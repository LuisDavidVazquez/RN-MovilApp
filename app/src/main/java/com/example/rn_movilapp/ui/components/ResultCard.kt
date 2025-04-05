package com.example.rn_movilapp.ui.components

import android.content.Context
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.example.rn_movilapp.data.ConnectorInfo
import com.example.rn_movilapp.ui.theme.TextPrimary
import com.example.rn_movilapp.ui.theme.TextSecondary

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultCard(
    connectorType: String,
    confidence: Float,
    modifier: Modifier = Modifier,
    showOtherPossibilities: Boolean = false
) {
    val context = LocalContext.current
    var connectorInfo by remember { mutableStateOf<ConnectorInfo?>(null) }
    var isExpanded by remember { mutableStateOf(false) }
    
    // Animación para la barra de progreso
    val animatedProgress by animateFloatAsState(
        targetValue = confidence,
        animationSpec = tween(1000),
        label = "progress"
    )

    LaunchedEffect(connectorType) {
        connectorInfo = getConnectorInfo(context, connectorType)
    }

    ElevatedCard(
        modifier = modifier,
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "Resultado",
                style = MaterialTheme.typography.headlineMedium.copy(
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Bold
                ),
                modifier = Modifier.padding(bottom = 16.dp)
            )

            // Conector detectado con icono
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Cable,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "Conector detectado:",
                        style = MaterialTheme.typography.bodyLarge.copy(
                            color = TextSecondary
                        )
                    )
                }
                Text(
                    text = connectorType,
                    style = MaterialTheme.typography.bodyLarge.copy(
                        color = TextPrimary,
                        fontWeight = FontWeight.Medium
                    )
                )
            }

            // Barra de progreso con porcentaje
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Analytics,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "Confianza:",
                        style = MaterialTheme.typography.bodyLarge.copy(
                            color = TextSecondary
                        )
                    )
                }
                
                Box(
                    modifier = Modifier
                        .weight(2f)
                        .height(8.dp)
                        .clip(MaterialTheme.shapes.small)
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxHeight()
                            .fillMaxWidth(animatedProgress)
                            .clip(MaterialTheme.shapes.small)
                            .background(MaterialTheme.colorScheme.primary)
                    )
                }
                
                Text(
                    text = "${(confidence * 100).toInt()}%",
                    style = MaterialTheme.typography.bodyLarge.copy(
                        color = TextPrimary,
                        fontWeight = FontWeight.Medium
                    ),
                    modifier = Modifier.width(48.dp)
                )
            }

            // Información técnica
            if (connectorInfo != null) {
                Divider(
                    modifier = Modifier
                        .padding(vertical = 16.dp)
                        .fillMaxWidth(),
                    color = MaterialTheme.colorScheme.outlineVariant
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Información Técnica",
                        style = MaterialTheme.typography.titleMedium.copy(
                            color = TextPrimary,
                            fontWeight = FontWeight.Medium
                        )
                    )
                    IconButton(onClick = { isExpanded = !isExpanded }) {
                        Icon(
                            imageVector = if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                            contentDescription = if (isExpanded) "Mostrar menos" else "Mostrar más"
                        )
                    }
                }

                if (isExpanded) {
                    TechnicalInfoItem(
                        icon = Icons.Default.DeviceHub,
                        title = "Compatibilidad",
                        content = connectorInfo!!.compatibility
                    )
                    TechnicalInfoItem(
                        icon = Icons.Default.Speed,
                        title = "Velocidad",
                        content = connectorInfo!!.speed
                    )
                    TechnicalInfoItem(
                        icon = Icons.Default.Power,
                        title = "Energía",
                        content = connectorInfo!!.power
                    )
                    TechnicalInfoItem(
                        icon = Icons.Default.Info,
                        title = "Usos",
                        content = connectorInfo!!.uses
                    )
                }
            }
        }
    }
}

@Composable
private fun TechnicalInfoItem(
    icon: ImageVector,
    title: String,
    content: String,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(20.dp)
            )
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall.copy(
                    color = TextSecondary
                )
            )
        }
        Text(
            text = content,
            style = MaterialTheme.typography.bodyMedium.copy(
                color = TextPrimary
            ),
            modifier = Modifier.padding(start = 28.dp, top = 4.dp)
        )
    }
}

private fun getConnectorInfo(context: Context, connectorType: String): ConnectorInfo? {
    return try {
        val jsonString = context.assets.open("info.json").bufferedReader().use { it.readText() }
        val typeToken = object : TypeToken<Map<String, ConnectorInfo>>() {}.type
        val connectorMap: Map<String, ConnectorInfo> = Gson().fromJson(jsonString, typeToken)
        connectorMap[connectorType]
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
} 