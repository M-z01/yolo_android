package com.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/** A detected object. [box] is normalized to [0, 1] in the upright (rotation-corrected) frame. */
data class Detection(val box: RectF, val classId: Int, val label: String, val score: Float)

/** Per-stage wall time of one [OnnxDetector.detect] call, in milliseconds. */
data class StageTimings(val preprocessMs: Float, val inferenceMs: Float, val postprocessMs: Float)

/**
 * Runs an Ultralytics YOLO ONNX export. Handles both output layouts:
 *  - end-to-end / NMS-free (YOLO26 default): [1, N, 6] rows of x1, y1, x2, y2, score, class
 *  - raw (YOLOv8/YOLO11): [1, 4 + numClasses, N] columns of cx, cy, w, h, class scores — needs NMS
 *
 * Not thread-safe: call [detect] and [close] from a single thread.
 *
 * @param model the .onnx file contents
 */
class OnnxDetector(
    model: ByteArray,
    var confThreshold: Float = 0.25f,
    var iouThreshold: Float = 0.45f,
) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()

    /** Execution provider actually in use, for display. */
    var executionProvider = "CPU"
        private set

    /** Timings of the most recent [detect] call. */
    var lastTimings = StageTimings(0f, 0f, 0f)
        private set

    private val session: OrtSession
    private val inputName: String
    private val inputWidth: Int
    private val inputHeight: Int
    private val labels: List<String>

    private val inputBitmap: Bitmap
    private val inputCanvas: Canvas
    private val pixels: IntArray
    private val chw: FloatArray
    private val inputBuffer: FloatBuffer
    private val matrix = Matrix()
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private var output = FloatArray(0)

    init {
        session = createSession(model)
        inputName = session.inputNames.first()
        val shape = (session.inputInfo.getValue(inputName).info as TensorInfo).shape // [1, 3, H, W]
        inputHeight = shape[2].toInt().takeIf { it > 0 } ?: DEFAULT_INPUT_SIZE
        inputWidth = shape[3].toInt().takeIf { it > 0 } ?: DEFAULT_INPUT_SIZE
        // Ultralytics embeds class names in the model metadata, so labels can't drift from the model.
        labels = parseNames(session.metadata.customMetadata["names"])

        inputBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        inputCanvas = Canvas(inputBitmap)
        pixels = IntArray(inputWidth * inputHeight)
        chw = FloatArray(3 * inputWidth * inputHeight)
        inputBuffer = ByteBuffer.allocateDirect(chw.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        Log.i(TAG, "Loaded model: input ${inputWidth}x$inputHeight, ${labels.size} classes, $executionProvider")
    }

    private fun createSession(model: ByteArray): OrtSession {
        // Leave two cores for the camera and UI; using all of them was ~2x slower on a Snapdragon 8 Elite.
        val threads = (Runtime.getRuntime().availableProcessors() - 2).coerceIn(2, 6)
        try {
            OrtSession.SessionOptions().use { opts ->
                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                // Ops XNNPACK doesn't support fall back to ORT's own thread pool, so it needs
                // threads too; a single-threaded ORT pool made inference ~2x slower. Spinning is off so
                // the two pools don't burn each other's cores.
                opts.setIntraOpNumThreads(threads)
                opts.addConfigEntry("session.intra_op.allow_spinning", "0")
                opts.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                return env.createSession(model, opts).also { executionProvider = "XNNPACK" }
            }
        } catch (e: Exception) {
            Log.w(TAG, "XNNPACK unavailable, falling back to default CPU provider", e)
        }
        OrtSession.SessionOptions().use { opts ->
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(threads)
            executionProvider = "CPU"
            return env.createSession(model, opts)
        }
    }

    /** Detects objects in [frame], which the camera delivered rotated by [rotationDegrees]. */
    fun detect(frame: Bitmap, rotationDegrees: Int): List<Detection> {
        val start = SystemClock.elapsedRealtimeNanos()
        val sideways = rotationDegrees % 180 != 0
        val imageWidth = if (sideways) frame.height else frame.width
        val imageHeight = if (sideways) frame.width else frame.height
        val scale = min(inputWidth.toFloat() / imageWidth, inputHeight.toFloat() / imageHeight)
        val padX = (inputWidth - imageWidth * scale) / 2f
        val padY = (inputHeight - imageHeight * scale) / 2f

        // Rotate upright and letterbox in a single draw: center at origin, rotate, scale, re-center.
        matrix.reset()
        matrix.postTranslate(-frame.width / 2f, -frame.height / 2f)
        matrix.postRotate(rotationDegrees.toFloat())
        matrix.postScale(scale, scale)
        matrix.postTranslate(inputWidth / 2f, inputHeight / 2f)
        inputCanvas.drawColor(LETTERBOX_COLOR)
        inputCanvas.drawBitmap(frame, matrix, bitmapPaint)

        // ARGB pixels -> planar RGB floats in [0, 1].
        inputBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        val area = pixels.size
        for (i in 0 until area) {
            val p = pixels[i]
            chw[i] = ((p shr 16) and 0xFF) / 255f
            chw[area + i] = ((p shr 8) and 0xFF) / 255f
            chw[2 * area + i] = (p and 0xFF) / 255f
        }
        inputBuffer.rewind()
        inputBuffer.put(chw)
        inputBuffer.rewind()

        val inputShape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        val preprocessed = SystemClock.elapsedRealtimeNanos()
        var inferred = 0L
        val detections = OnnxTensor.createTensor(env, inputBuffer, inputShape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                inferred = SystemClock.elapsedRealtimeNanos()
                val tensor = result.get(0) as OnnxTensor
                val shape = tensor.info.shape
                val buffer = tensor.floatBuffer
                if (output.size != buffer.remaining()) output = FloatArray(buffer.remaining())
                buffer.get(output)
                if (shape.size == 3 && shape[2] == 6L) {
                    decodeEndToEnd(shape[1].toInt())
                } else {
                    nonMaxSuppression(decodeRaw(shape[1].toInt(), shape[2].toInt()))
                }
            }
        }

        // Model-input pixels -> normalized upright-image coordinates.
        val mapped = detections.map { d ->
            val b = d.box
            d.copy(
                box = RectF(
                    ((b.left - padX) / scale / imageWidth).coerceIn(0f, 1f),
                    ((b.top - padY) / scale / imageHeight).coerceIn(0f, 1f),
                    ((b.right - padX) / scale / imageWidth).coerceIn(0f, 1f),
                    ((b.bottom - padY) / scale / imageHeight).coerceIn(0f, 1f),
                )
            )
        }
        val end = SystemClock.elapsedRealtimeNanos()
        lastTimings = StageTimings(
            (preprocessed - start) / 1e6f, (inferred - preprocessed) / 1e6f, (end - inferred) / 1e6f
        )
        return mapped
    }

    private fun decodeEndToEnd(rows: Int): List<Detection> {
        val detections = ArrayList<Detection>()
        for (i in 0 until rows) {
            val o = i * 6
            val score = output[o + 4]
            if (score < confThreshold) continue
            val classId = output[o + 5].toInt()
            detections += Detection(
                RectF(output[o], output[o + 1], output[o + 2], output[o + 3]), classId, label(classId), score
            )
        }
        return detections.sortedByDescending { it.score }.take(MAX_DETECTIONS)
    }

    private fun decodeRaw(channels: Int, anchors: Int): List<Detection> {
        // Walk class-major so reads are sequential in memory.
        val bestScore = FloatArray(anchors)
        val bestClass = IntArray(anchors)
        for (c in 0 until channels - 4) {
            val rowStart = (4 + c) * anchors
            for (i in 0 until anchors) {
                val s = output[rowStart + i]
                if (s > bestScore[i]) {
                    bestScore[i] = s
                    bestClass[i] = c
                }
            }
        }
        val detections = ArrayList<Detection>()
        for (i in 0 until anchors) {
            if (bestScore[i] < confThreshold) continue
            val cx = output[i]
            val cy = output[anchors + i]
            val halfW = output[2 * anchors + i] / 2f
            val halfH = output[3 * anchors + i] / 2f
            val classId = bestClass[i]
            detections += Detection(
                RectF(cx - halfW, cy - halfH, cx + halfW, cy + halfH), classId, label(classId), bestScore[i]
            )
        }
        return detections
    }

    /** Greedy per-class NMS, matching Ultralytics' default (non-agnostic) behavior. */
    private fun nonMaxSuppression(candidates: List<Detection>): List<Detection> {
        val kept = ArrayList<Detection>()
        for (d in candidates.sortedByDescending { it.score }.take(MAX_NMS_CANDIDATES)) {
            if (kept.size >= MAX_DETECTIONS) break
            if (kept.none { it.classId == d.classId && iou(it.box, d.box) > iouThreshold }) kept += d
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val w = min(a.right, b.right) - max(a.left, b.left)
        val h = min(a.bottom, b.bottom) - max(a.top, b.top)
        if (w <= 0f || h <= 0f) return 0f
        val inter = w * h
        return inter / (a.width() * a.height() + b.width() * b.height() - inter)
    }

    private fun label(classId: Int) = labels.getOrElse(classId) { "class $classId" }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "OnnxDetector"
        private const val DEFAULT_INPUT_SIZE = 640
        private const val MAX_DETECTIONS = 100
        private const val MAX_NMS_CANDIDATES = 300
        private val LETTERBOX_COLOR = Color.rgb(114, 114, 114) // Ultralytics' padding value

        // Matches the Python dict repr Ultralytics stores, e.g. {0: 'person', 1: "o'clock"}.
        private val NAME_PATTERN = Regex("""(\d+)\s*:\s*(?:'([^']*)'|"([^"]*)")""")

        private fun parseNames(raw: String?): List<String> {
            if (raw == null) return emptyList()
            val byId = NAME_PATTERN.findAll(raw).associate { m ->
                m.groupValues[1].toInt() to m.groupValues[2].ifEmpty { m.groupValues[3] }
            }
            val count = (byId.keys.maxOrNull() ?: -1) + 1
            return List(count) { byId[it] ?: "class $it" }
        }
    }
}
