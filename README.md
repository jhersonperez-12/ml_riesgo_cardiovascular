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
│   ├── modelo_rf.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.json
└── data/
    └── Clasificacion_RCV_Completo.xlsx   ← coloca tu archivo aquí
```

## Instalación

```bash
pip install -r requirements.txt
```

## Paso 1 — Entrenar el modelo

Coloca el archivo Excel en la carpeta `data/` y ejecuta:

```bash
python src/train.py --data data/Clasificacion_RCV_Completo.xlsx
```

### Opciones disponibles

| Parámetro       | Descripción                              | Default               |
|-----------------|------------------------------------------|-----------------------|
| `--data`        | Ruta al archivo Excel (obligatorio)      | —                     |
| `--target`      | Nombre de la columna objetivo            | `CLASIFICACION_RIESGO`|
| `--n_iter`      | Iteraciones del RandomizedSearchCV       | `30`                  |
| `--models_dir`  | Carpeta de salida para artefactos        | `models`              |

Ejemplo rápido (menos iteraciones para prueba):

```bash
python src/train.py --data data/Clasificacion_RCV_Completo.xlsx --n_iter 10
```

## Paso 2 — Lanzar la aplicación

```bash
streamlit run app.py
```

La app abrirá automáticamente en `http://localhost:8501`.

## Flujo del modelo

```
Excel → Limpieza de valores inválidos (99, 999, 9999, 99999)
      → Imputación por mediana
      → Capeo de outliers (IQR / percentil 99)
      → One-hot encoding de variables categóricas
      → StandardScaler
      → RandomForestClassifier (optimizado con RandomizedSearchCV)
      → Clasificación: ALTO / MEDIO / BAJO
```

## Métrica objetivo

El modelo optimiza el **Recall de la clase ALTO**, priorizando que los pacientes
de alto riesgo sean correctamente identificados.

## Notas

- Los artefactos (`modelo_rf.pkl`, `scaler.pkl`, `label_encoder.pkl`, `feature_names.json`)
  deben estar en la carpeta `models/` antes de lanzar la app.
- Las opciones de las variables categóricas en la app (`SEXO`, `TABAQUISMO`, etc.)
  deben coincidir con los valores reales de tu dataset. Ajústalas en `app.py` si es necesario.
