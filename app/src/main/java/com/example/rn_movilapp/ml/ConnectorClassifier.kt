package com.example.rn_movilapp.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier

class ConnectorClassifier(private val context: Context) {
    private var classifier: ImageClassifier? = null
    private val classNames = listOf(
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
        try {
            val options = ImageClassifier.ImageClassifierOptions.builder()
                .setMaxResults(5)
                .build()

            classifier = ImageClassifier.createFromFileAndOptions(
                context,
                "mejor_modelo_ft.tflite",
                options
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun classify(bitmap: Bitmap): List<Category>? {
        if (classifier == null) return null

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
        val results = classifier?.classify(tensorImage)

        return results?.firstOrNull()?.categories?.map { category ->
            Category(
                classNames[category.index],
                category.score
            )
        }
    }
} 