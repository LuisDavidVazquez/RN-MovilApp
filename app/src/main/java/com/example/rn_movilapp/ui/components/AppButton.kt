package com.example.rn_movilapp.ui.components

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun AppButton(
    text: String,
    onClick: () -> Unit,
    isPrimary: Boolean = true,
    modifier: Modifier = Modifier
) {
    if (isPrimary) {
        Button(
            onClick = onClick,
            modifier = modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
                .height(48.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = MaterialTheme.colorScheme.onPrimary
            ),
            elevation = ButtonDefaults.buttonElevation(
                defaultElevation = 2.dp,
                pressedElevation = 8.dp
            )
        ) {
            Text(
                text = text,
                style = MaterialTheme.typography.bodyLarge.copy(
                    fontWeight = FontWeight.Medium
                )
            )
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            modifier = modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
                .height(48.dp),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = MaterialTheme.colorScheme.primary
            )
        ) {
            Text(
                text = text,
                style = MaterialTheme.typography.bodyLarge.copy(
                    fontWeight = FontWeight.Medium
                )
            )
        }
    }
} 