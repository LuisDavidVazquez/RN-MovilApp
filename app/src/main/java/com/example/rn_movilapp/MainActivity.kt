package com.example.rn_movilapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.rn_movilapp.ui.components.AppButton
import com.example.rn_movilapp.ui.components.ResultCard
import com.example.rn_movilapp.ui.theme.RNMovilAppTheme
import android.provider.MediaStore
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.layout.ContentScale

import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.compose.ui.graphics.Color


class MainActivity : ComponentActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }

    private var hasPermission by mutableStateOf(false)

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasPermission = isGranted
        if (isGranted) {
            Log.d(TAG, "Permiso de cámara concedido")
            Toast.makeText(this, "Permiso de cámara concedido", Toast.LENGTH_SHORT).show()
        } else {
            Log.e(TAG, "Permiso de cámara denegado")
            Toast.makeText(this, "Se requiere permiso de cámara", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate iniciado")

        // Verificar permisos de cámara
        hasPermission = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        setContent {
            RNMovilAppTheme {
                var showCamera by remember { mutableStateOf(false) }
                var classificationResult by remember { mutableStateOf<ClassificationResult?>(null) }
                var isAnalyzing by remember { mutableStateOf(false) }
                val context = LocalContext.current
                var selectedImageBitmap by remember { mutableStateOf<Bitmap?>(null) }

                // Launcher para seleccionar imagen
                val imagePickerLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.GetContent()
                ) { uri: Uri? ->
                    if (uri == null) {
                        Log.e(TAG, "URI es null - No se seleccionó ninguna imagen")
                        Toast.makeText(context, "No se seleccionó ninguna imagen", Toast.LENGTH_SHORT).show()
                        return@rememberLauncherForActivityResult
                    }

                    try {
                        Log.d(TAG, "Iniciando carga de imagen desde URI: $uri")

                        selectedImageBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                            val source = ImageDecoder.createSource(context.contentResolver, uri)
                            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                                decoder.isMutableRequired = true
                            }
                        } else {
                            @Suppress("DEPRECATION")
                            MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                        }

                        if (selectedImageBitmap == null) {
                            Log.e(TAG, "selectedImageBitmap es null después de la decodificación")
                            Toast.makeText(context, "Error al procesar la imagen", Toast.LENGTH_SHORT).show()
                            return@rememberLauncherForActivityResult
                        }

                        Log.d(TAG, "Imagen cargada exitosamente. Dimensiones: ${selectedImageBitmap!!.width}x${selectedImageBitmap!!.height}")

                        // Clasificar la imagen seleccionada
                        val classifier = ConnectorClassifier(context)
                        classificationResult = classifier.classify(selectedImageBitmap!!)
                        classifier.close()
                        isAnalyzing = true
                        showCamera = false

                        Log.d(TAG, "Clasificación completada. Resultado: ${classificationResult?.className}")

                    } catch (e: Exception) {
                        Log.e(TAG, "Error al cargar/procesar la imagen", e)
                        val errorMessage = when (e) {
                            is SecurityException -> "Error de permisos al acceder a la imagen"
                            is OutOfMemoryError -> "La imagen es demasiado grande para procesarla"
                            else -> "Error al cargar la imagen: ${e.message}"
                        }
                        Toast.makeText(context, errorMessage, Toast.LENGTH_LONG).show()
                        selectedImageBitmap = null
                    }
                }

                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {

                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(16.dp)
                            ) {


                                // Título personalizado
                                Text(
                                    text = "Conetec",
                                    style = MaterialTheme.typography.headlineMedium.copy(
                                        fontSize = 32.sp,         // tamaño en sp
                                        fontWeight = FontWeight.Bold,
                                        color = Color(0xFF1E88E5) // azul personalizado (puedes usar Color.Red, Color.White, etc.)
                                    ),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(bottom = 8.dp),
                                    textAlign = TextAlign.Center
                                )

// Descripción personalizada
                                Text(
                                    text = "Clasificador de Objetos para la Normatización de Estándares y\n" +
                                            "Conectores Tecnológicos Eléctricos y Computacionales",
                                    style = MaterialTheme.typography.titleMedium.copy(
                                        fontSize = 18.sp,
                                        fontWeight = FontWeight.Medium,
                                        color = Color.DarkGray
                                    ),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(bottom = 16.dp),
                                    textAlign = TextAlign.Center
                                )


                                if (!showCamera && selectedImageBitmap == null) {
                            // Instrucciones
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 16.dp)
                            ) {
                                Column(
                                    modifier = Modifier
                                        .padding(16.dp)
                                ) {
                                    Text(
                                        text = "Instrucciones:",
                                        style = MaterialTheme.typography.titleMedium
                                    )
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text("1. Haz clic en 'Iniciar Cámara' para comenzar.")
                                    Text("2. Coloca el conector frente a la cámara.")
                                    Text("3. Presiona 'Analizar' para identificar el conector.")
                                    Text("4. También puedes importar una imagen existente.")
                                }
                            }
                        }

                        if (showCamera) {
                            Column(
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxWidth()
                            ) {
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .fillMaxWidth()
                                ) {
                                    CameraView(
                                        onImageClassified = { result ->
                                            classificationResult = result
                                            isAnalyzing = false  // Detener el análisis después de obtener un resultado
                                        },
                                        shouldAnalyze = isAnalyzing
                                    )
                                }

                                // Mostrar el resultado sobre la cámara si existe
                                if (classificationResult != null) {
                                    ResultCard(
                                        connectorType = classificationResult!!.className,
                                        confidence = classificationResult!!.confidence,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(16.dp)
                                    )
                                }
                            }
                        } else if (selectedImageBitmap != null) {
                            Column(
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxWidth()
                            ) {
                                // Mostrar la imagen seleccionada
                                Image(
                                    bitmap = selectedImageBitmap!!.asImageBitmap(),
                                    contentDescription = "Imagen seleccionada",
                                    modifier = Modifier
                                        .weight(1f)
                                        .fillMaxWidth()
                                        .padding(16.dp)
                                )

                                // Mostrar el resultado si existe
                                if (classificationResult != null) {
                                    ResultCard(
                                        connectorType = classificationResult!!.className,
                                        confidence = classificationResult!!.confidence,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(16.dp)
                                    )
                                }

                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(16.dp),
                                    horizontalArrangement = Arrangement.SpaceEvenly
                                ) {
                                    if (classificationResult == null) {
                                        AppButton(
                                            text = "Analizar Imagen",
                                            onClick = {
                                                val classifier = ConnectorClassifier(context)
                                                classificationResult = classifier.classify(selectedImageBitmap!!)
                                                classifier.close()
                                            },
                                            isPrimary = true
                                        )
                                    } else {
                                        AppButton(
                                            text = "Nuevo Análisis",
                                            onClick = {
                                                classificationResult = null
                                                imagePickerLauncher.launch("image/*")
                                            },
                                            isPrimary = true
                                        )
                                    }

                                    AppButton(
                                        text = "Volver",
                                        onClick = {
                                            selectedImageBitmap = null
                                            classificationResult = null
                                        },
                                        isPrimary = false
                                    )
                                }
                            }
                        }

                        // Botones de control
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 16.dp)
                        ) {
                            if (!showCamera && selectedImageBitmap == null) {
                                AppButton(
                                    text = "Iniciar Cámara",
                                    onClick = {
                                        if (!hasPermission) {
                                            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
                                        } else {
                                            showCamera = true
                                        }
                                    },
                                    isPrimary = true
                                )

                                AppButton(
                                    text = "Importar Imagen",
                                    onClick = { imagePickerLauncher.launch("image/*") },
                                    isPrimary = false
                                )
                            } else if (showCamera) {
                                if (classificationResult == null) {
                                    AppButton(
                                        text = "Analizar",
                                        onClick = { isAnalyzing = true },
                                        isPrimary = true
                                    )
                                } else {
                                    AppButton(
                                        text = "Nuevo Análisis",
                                        onClick = {
                                            classificationResult = null
                                            isAnalyzing = false
                                        },
                                        isPrimary = true
                                    )
                                }

                                AppButton(
                                    text = "Detener Cámara",
                                    onClick = {
                                        showCamera = false
                                        isAnalyzing = false
                                        classificationResult = null
                                    },
                                    isPrimary = false
                                )
                            } else if (selectedImageBitmap != null) {
                                AppButton(
                                    text = "Volver",
                                    onClick = {
                                        selectedImageBitmap = null
                                        isAnalyzing = false
                                        classificationResult = null
                                    },
                                    isPrimary = false
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}