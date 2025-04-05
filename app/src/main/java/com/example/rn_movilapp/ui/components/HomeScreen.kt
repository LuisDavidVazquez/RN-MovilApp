package com.example.rn_movilapp.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.rn_movilapp.R

@Composable
fun HomeScreen(
    onCameraClick: () -> Unit,
    onImportClick: () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Logo y título
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 24.dp),
                contentAlignment = Alignment.Center
            ) {
                Image(
                    painter = painterResource(id = R.drawable.icon),
                    contentDescription = "Logo CONECTEC",
                    modifier = Modifier
                        .size(180.dp)
                        .clip(RoundedCornerShape(16.dp)),
                    contentScale = ContentScale.Fit
                )
            }

            // Título principal
            Text(
                text = "CONECTEC",
                style = MaterialTheme.typography.headlineLarge.copy(
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                ),
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            // Subtítulo descriptivo
            Text(
                text = "Clasificador de Objetos para\n" +
                       "la Normatización de Estándares y\n" +
                       "Conectores Tecnológicos Eléctricos y\n" +
                       "Computacionales",
                style = MaterialTheme.typography.titleMedium.copy(
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
                ),
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(bottom = 32.dp)
            )

            // Tarjeta de instrucciones
            ElevatedCard(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 24.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Text(
                        text = "Instrucciones",
                        style = MaterialTheme.typography.titleLarge.copy(
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold
                        ),
                        modifier = Modifier.padding(bottom = 16.dp)
                    )

                    InstructionItem(
                        number = "1",
                        text = "Haz clic en \"Iniciar Cámara\" para activar la cámara."
                    )
                    InstructionItem(
                        number = "2",
                        text = "Coloca el conector/adaptador frente a la cámara."
                    )
                    InstructionItem(
                        number = "3",
                        text = "Presiona \"Analizar\" para identificar en tiempo real."
                    )
                    InstructionItem(
                        number = "4",
                        text = "Alternativamente, usa \"Importar Imagen\" para analizar una foto existente."
                    )
                }
            }

            // Botones de acción
            AppButton(
                text = "Iniciar Cámara",
                onClick = onCameraClick,
                isPrimary = true
            )

            AppButton(
                text = "Importar Imagen",
                onClick = onImportClick,
                isPrimary = false
            )
        }
    }
}

@Composable
private fun InstructionItem(
    number: String,
    text: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Círculo con número
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.1f)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = number,
                style = MaterialTheme.typography.titleMedium.copy(
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold
                )
            )
        }

        // Texto de la instrucción
        Text(
            text = text,
            style = MaterialTheme.typography.bodyLarge.copy(
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
            )
        )
    }
} 