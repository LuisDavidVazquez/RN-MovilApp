package com.example.rn_movilapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import android.graphics.BitmapFactory
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream
import android.graphics.ImageFormat

private const val TAG = "CameraView"

@Composable
fun CameraView(
    onImageClassified: (ClassificationResult) -> Unit,
    shouldAnalyze: Boolean = false
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    var lastAnalyzedImage by remember { mutableStateOf<ImageProxy?>(null) }

    DisposableEffect(lifecycleOwner) {
        onDispose {
            lastAnalyzedImage?.close()
        }
    }

    AndroidView(
        factory = { ctx ->
            PreviewView(ctx).apply {
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            }
        },
        modifier = Modifier.fillMaxSize()
    ) { previewView ->
        val executor = ContextCompat.getMainExecutor(context)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build()
            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .apply {
                    setAnalyzer(executor) { imageProxy ->
                        if (shouldAnalyze) {
                            try {
                                val bitmap = imageProxy.toBitmap()
                                val classifier = ConnectorClassifier(context)
                                val result = classifier.classify(bitmap)
                                classifier.close()
                                onImageClassified(result)
                                lastAnalyzedImage?.close()
                                lastAnalyzedImage = imageProxy
                            } catch (e: Exception) {
                                Log.e("CameraView", "Error al analizar imagen", e)
                                imageProxy.close()
                            }
                        } else {
                            imageProxy.close()
                        }
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
                preview.setSurfaceProvider(previewView.surfaceProvider)
            } catch (e: Exception) {
                Log.e("CameraView", "Error al vincular la cámara", e)
            }
        }, executor)
    }
}

private fun ImageProxy.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
} 