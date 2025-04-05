package com.example.rn_movilapp.data

data class ConnectorInfo(
    val compatibility: String,
    val speed: String,
    val power: String,
    val uses: String
)

data class ConnectorInfoMap(
    val connectors: Map<String, ConnectorInfo>
) 