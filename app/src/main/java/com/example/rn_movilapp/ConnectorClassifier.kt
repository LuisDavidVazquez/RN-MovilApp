package com.example.rn_movilapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ConnectorClassifier(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val modelPath = "mejor_modelo_ft.tflite"
    private val imageSize = 224 // Tamaño de imagen requerido
    private val numClasses = 20 // Número de clases en el modelo
    
    private val classNames = arrayOf(
        "Conector Lightning (Apple)",
        "Cable Audio Óptico",
        "Adaptador de corriente (clavija redonda)",
        "Clavija US (Americana)",
        "Cable Coaxial",
        "Adaptador de corriente 6 salidas",
        "Adaptador de corriente (clavija redonda)",
        "Conector DisplayPort",
        "Conector HDMI",
        "Adaptador jack de audio de 3.5mm",
        "Cargador magnetico",
        "Conector Micro HDMI",
        "Conector Micro-USB",
        "Adaptador de corriente multicontacto",
        "Conector RCA",
        "Conector RJ-45 (Ethernet)",
        "Adaptador multipuerto USB hub",
        "Conector USB tipo A",
        "Conector USB tipo C",
        "Conector VGA"
    )
    
    init {
        loadModel()
    }

    private fun loadModel() {
        val modelBuffer = loadModelFile()
        val options = Interpreter.Options()
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        // Preprocesar la imagen igual que en la aplicación web
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalizar a [0,1]
            .build()

        var tensorImage = TensorImage.fromBitmap(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // Preparar el buffer de salida para todas las clases
        val outputBuffer = Array(1) { FloatArray(numClasses) }

        // Ejecutar la inferencia
        interpreter?.run(tensorImage.buffer, outputBuffer)

        // Encontrar la clase con mayor probabilidad
        var maxIdx = 0
        var maxProb = outputBuffer[0][0]
        for (i in 1 until numClasses) {
            if (outputBuffer[0][i] > maxProb) {
                maxIdx = i
                maxProb = outputBuffer[0][i]
            }
        }

        return ClassificationResult(
            className = classNames[maxIdx],
            confidence = maxProb
        )
    }

    fun close() {
        interpreter?.close()
    }
}

data class ClassificationResult(
    val className: String,
    val confidence: Float
) 