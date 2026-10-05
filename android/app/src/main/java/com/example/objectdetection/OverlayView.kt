package com.example.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import kotlin.math.max
import kotlin.math.roundToInt

/** Draws detection boxes over a PreviewView using its default FILL_CENTER scaling. */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var imageWidth = 1
    private var imageHeight = 1

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2.5f)
    }
    private val labelBackgroundPaint = Paint().apply { style = Paint.Style.FILL }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 14f, resources.displayMetrics)
    }
    private val textMetrics = textPaint.fontMetrics
    private val labelPadding = dp(4f)
    private val rect = RectF()

    /** [imageWidth]/[imageHeight] are the upright analysis frame size the boxes are normalized to. */
    fun setResults(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        this.detections = detections
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // FILL_CENTER scales the frame to cover the view and crops the overflow equally on both sides.
        val scale = max(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f
        val textHeight = textMetrics.descent - textMetrics.ascent

        for (d in detections) {
            rect.set(
                d.box.left * imageWidth * scale + offsetX,
                d.box.top * imageHeight * scale + offsetY,
                d.box.right * imageWidth * scale + offsetX,
                d.box.bottom * imageHeight * scale + offsetY,
            )
            val color = PALETTE[d.classId % PALETTE.size]
            boxPaint.color = color
            labelBackgroundPaint.color = color
            canvas.drawRect(rect, boxPaint)

            val text = "${d.label} ${(d.score * 100).roundToInt()}%"
            val labelTop = (rect.top - textHeight).coerceAtLeast(0f)
            canvas.drawRect(
                rect.left, labelTop,
                rect.left + textPaint.measureText(text) + 2 * labelPadding, labelTop + textHeight,
                labelBackgroundPaint,
            )
            canvas.drawText(text, rect.left + labelPadding, labelTop - textMetrics.ascent, textPaint)
        }
    }

    private fun dp(value: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value, resources.displayMetrics)

    companion object {
        private val PALETTE = intArrayOf(
            0xFFFF3838.toInt(), 0xFF2C99A8.toInt(), 0xFFFF9D97.toInt(), 0xFF00C2FF.toInt(),
            0xFFFF701F.toInt(), 0xFF344593.toInt(), 0xFFFFB21D.toInt(), 0xFF6473FF.toInt(),
            0xFFCFD231.toInt(), 0xFF0018EC.toInt(), 0xFF48F90A.toInt(), 0xFF8438FF.toInt(),
            0xFF92CC17.toInt(), 0xFF520085.toInt(), 0xFF3DDB86.toInt(), 0xFFCB38FF.toInt(),
        )
    }
}
