# 🫀 Clasificador de Riesgo Cardiovascular

Proyecto de Machine Learning para clasificación de riesgo cardiovascular usando **Random Forest**.

## Estructura del proyecto

```
rcv_project/
├── app.py                  # Aplicación Streamlit
├── requirements.txt        # Dependencias
├── src/
│   ├── train.py            # Script de entrenamiento
│   └── preprocessing.py    # Módulo de preprocesamiento
├── models/                 # Artefactos generados al entrenar
│   ├── modelo_rf.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── feature_names.json
└── data/
    └── Clasificacion_RCV_Completo.xlsx   ← coloca tu archivo aquí
```