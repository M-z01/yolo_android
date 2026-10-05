package com.example.objectdetection

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updateLayoutParams
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var statsText: TextView

    private lateinit var modelStore: ModelStore

    // Model loading and inference share one thread, so the detector needs no locking.
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var detector: OnnxDetector? = null
    private var detectorName = ""
    private var imageAnalysis: ImageAnalysis? = null
    private var lastFrameNanos = 0L
    private var smoothedFps = 0f
    private val timingSums = FloatArray(4) // convert, pre, inference, post
    private var timedFrames = 0

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else statsText.setText(R.string.camera_permission_denied)
        }

    private val pickModelFile =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let(::importModel) }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview_view)
        overlay = findViewById(R.id.overlay)
        statsText = findViewById(R.id.stats_text)
        statsText.setOnClickListener { showModelPicker() }
        modelStore = ModelStore(this)

        val baseMargin = (statsText.layoutParams as ViewGroup.MarginLayoutParams).topMargin
        ViewCompat.setOnApplyWindowInsetsListener(statsText) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.updateLayoutParams<ViewGroup.MarginLayoutParams> {
                topMargin = baseMargin + bars.top
                leftMargin = baseMargin + bars.left
            }
            insets
        }

        analysisExecutor.execute {
            val model = modelStore.selected()
            if (!loadModel(model.displayName) { modelStore.read(model) } && model != ModelStore.BUILT_IN) {
                loadModel(ModelStore.BUILT_IN.displayName) { modelStore.read(ModelStore.BUILT_IN) }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    /**
     * Swaps in a detector for [name]; on failure the current one keeps running. Analysis thread only,
     * so frames never see a half-switched detector.
     */
    private fun loadModel(name: String, bytes: () -> ByteArray): Boolean {
        val loaded = try {
            OnnxDetector(bytes())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model $name", e)
            showMessage(getString(R.string.model_load_failed, name, e.message))
            return false
        }
        detector?.close()
        detector = loaded
        detectorName = name
        smoothedFps = 0f
        lastFrameNanos = 0L
        return true
    }

    private fun showModelPicker() {
        val models = modelStore.list()
        val current = models.indexOfFirst { it.id == modelStore.selected().id }
        AlertDialog.Builder(this)
            .setTitle(R.string.choose_model)
            .setSingleChoiceItems(models.map { it.displayName }.toTypedArray(), current) { dialog, which ->
                dialog.dismiss()
                val model = models[which]
                analysisExecutor.execute {
                    if (loadModel(model.displayName) { modelStore.read(model) }) modelStore.select(model)
                }
            }
            .setNeutralButton(R.string.import_model) { _, _ -> pickModelFile.launch(arrayOf("*/*")) }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    /** Copies a model from the system file picker into the model folder, if it loads. */
    private fun importModel(uri: Uri) = analysisExecutor.execute {
        val (fileName, bytes) = try {
            modelStore.readDocument(uri)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read $uri", e)
            showMessage(getString(R.string.model_import_failed, e.message))
            return@execute
        }
        // Load before saving so a bad file never replaces a working model of the same name.
        if (loadModel(fileName.removeSuffix(".onnx")) { bytes }) {
            val model = modelStore.save(fileName, bytes)
            modelStore.select(model)
            showMessage(getString(R.string.model_imported, model.displayName))
        }
    }

    private fun showMessage(message: String) = runOnUiThread {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            // Preview and analysis share an aspect ratio so overlay boxes line up with the preview.
            val preview = Preview.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .build()
                )
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            // Small analysis frames: the model input is 640 px at most, so larger frames only cost time.
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(640, 480),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            )
                        )
                        .build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::analyze) }
            imageAnalysis = analysis

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        val detector = detector
        if (detector == null) {
            image.close()
            return
        }
        val start = SystemClock.elapsedRealtimeNanos()
        val rotation = image.imageInfo.rotationDegrees
        val frame = image.toBitmap()
        image.close()
        val convertMs = (SystemClock.elapsedRealtimeNanos() - start) / 1e6f

        val detections = detector.detect(frame, rotation)
        val timings = detector.lastTimings
        val now = SystemClock.elapsedRealtimeNanos()
        logAverageTimings(convertMs, timings)
        if (lastFrameNanos != 0L) {
            val fps = 1e9f / (now - lastFrameNanos)
            smoothedFps = if (smoothedFps == 0f) fps else 0.9f * smoothedFps + 0.1f * fps
        }
        lastFrameNanos = now

        val sideways = rotation % 180 != 0
        val uprightWidth = if (sideways) frame.height else frame.width
        val uprightHeight = if (sideways) frame.width else frame.height
        val stats = getString(
            R.string.stats, detectorName, detector.executionProvider, smoothedFps, detections.size,
            convertMs, timings.preprocessMs, timings.inferenceMs, timings.postprocessMs,
        )
        runOnUiThread {
            overlay.setResults(detections, uprightWidth, uprightHeight)
            statsText.text = stats
        }
    }

    private fun logAverageTimings(convertMs: Float, timings: StageTimings) {
        timingSums[0] += convertMs
        timingSums[1] += timings.preprocessMs
        timingSums[2] += timings.inferenceMs
        timingSums[3] += timings.postprocessMs
        if (++timedFrames < LOG_EVERY_FRAMES) return
        val (convert, pre, inference, post) = timingSums.map { it / timedFrames }
        Log.i(TAG, "avg ms: convert %.1f pre %.1f infer %.1f post %.1f | %.1f FPS".format(
            convert, pre, inference, post, smoothedFps))
        timingSums.fill(0f)
        timedFrames = 0
    }

    override fun onDestroy() {
        super.onDestroy()
        imageAnalysis?.clearAnalyzer()
        analysisExecutor.execute { detector?.close() }
        analysisExecutor.shutdown()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val LOG_EVERY_FRAMES = 30
    }
}
