package com.example.objectdetection

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.core.content.edit
import java.io.File

/**
 * The models the user can switch between: the one bundled in the APK plus any .onnx files in
 * [directory], which `export.py --push` writes to over adb and in-app imports copy into.
 */
class ModelStore(private val context: Context) {

    /** A selectable model; [file] is null for the bundled one. */
    data class Model(val id: String, val displayName: String, val file: File?)

    /**
     * /sdcard/Android/data/<package>/files/models. The app must create it: adb can then push files
     * into it, but the app can't read a folder that adb created.
     */
    val directory: File = requireNotNull(context.getExternalFilesDir("models")) { "Shared storage unavailable" }

    private val prefs = context.getSharedPreferences("models", Context.MODE_PRIVATE)

    fun list(): List<Model> {
        val files = directory.listFiles { f -> f.isFile && f.name.endsWith(ONNX) }.orEmpty()
        return listOf(BUILT_IN) + files.sortedBy { it.name.lowercase() }.map { Model(it.name, it.nameWithoutExtension, it) }
    }

    /** The remembered model, or the bundled one if its file has since been removed. */
    fun selected(): Model {
        val id = prefs.getString(KEY_SELECTED, null)
        return list().find { it.id == id } ?: BUILT_IN
    }

    fun select(model: Model) = prefs.edit { putString(KEY_SELECTED, model.id) }

    fun read(model: Model): ByteArray =
        model.file?.readBytes() ?: context.assets.open(BUILT_IN_ASSET).use { it.readBytes() }

    /** Reads a document chosen in the system file picker; returns its .onnx file name and contents. */
    fun readDocument(uri: Uri): Pair<String, ByteArray> {
        val name = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { if (it.moveToFirst()) it.getString(0) else null }
            ?: "imported-${System.currentTimeMillis()}"
        val fileName = name.replace('/', '_').let { if (it.endsWith(ONNX)) it else it + ONNX }
        val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: error("Can't open $uri")
        return fileName to bytes
    }

    /** Saves model contents into [directory], replacing a model with the same file name. */
    fun save(fileName: String, bytes: ByteArray): Model {
        val file = File(directory, fileName)
        file.writeBytes(bytes)
        return Model(file.name, file.nameWithoutExtension, file)
    }

    companion object {
        private const val ONNX = ".onnx"
        private const val BUILT_IN_ASSET = "model.onnx"
        private const val KEY_SELECTED = "selected"
        val BUILT_IN = Model("built-in", "Built-in", null)
    }
}
