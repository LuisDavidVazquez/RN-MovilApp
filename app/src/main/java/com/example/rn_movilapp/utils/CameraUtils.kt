package com.example.rn_movilapp.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

object CameraUtils {
    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val planeBuffer = image.planes[0].buffer
        val data = ByteArray(planeBuffer.remaining())
        planeBuffer.get(data)

        val imageWidth = image.width
        val imageHeight = image.height
        val pixelStride = image.planes[0].pixelStride
        val rowStride = image.planes[0].rowStride
        val rowPadding = rowStride - pixelStride * imageWidth

        val bitmap = Bitmap.createBitmap(
            imageWidth + rowPadding / pixelStride,
            imageHeight,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(data))
        return bitmap
    }

    fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }
        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }
} 