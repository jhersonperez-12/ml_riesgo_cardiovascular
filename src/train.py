"""
train.py — Entrena el modelo Random Forest para clasificación de riesgo cardiovascular.
Uso: python src/train.py --data data/Clasificacion_RCV_Completo.xlsx
"""

import argparse
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ──────────────────────────────────────────────
# 1. CARGA DE DATOS
# ──────────────────────────────────────────────

def cargar_datos(ruta: str) -> pd.DataFrame:
    df = pd.read_excel(ruta)
    print(f"✔ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# ──────────────────────────────────────────────
# 2. PREPROCESAMIENTO
# ──────────────────────────────────────────────

def preprocesar(df: pd.DataFrame):
    # Reemplazar valores inválidos
    valores_invalidos = [99999, 999, 9999, 99]
    df.replace(valores_invalidos, np.nan, inplace=True)

    # Imputar con mediana
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Capar outliers con IQR
    def capar_iqr(df, columna):
        Q1, Q3 = df[columna].quantile(0.25), df[columna].quantile(0.75)
        IQR = Q3 - Q1
        df[columna] = np.clip(df[columna], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return df

    for col in ['PESO', 'TALLA', 'EDAD', 'IMC']:
        if col in df.columns:
            df = capar_iqr(df, col)

    # Colesterol HDL mínimo
    if 'COLESTEROL_HDL' in df.columns:
        df['COLESTEROL_HDL'] = np.clip(df['COLESTEROL_HDL'], 20, None)

    # Colesterol LDL — reemplazar valores < 20 por mediana válida
    if 'COLESTEROL_LDL' in df.columns:
        mediana_ldl = df[df['COLESTEROL_LDL'] >= 20]['COLESTEROL_LDL'].median()
        df.loc[df['COLESTEROL_LDL'] < 20, 'COLESTEROL_LDL'] = mediana_ldl

    # Triglicéridos — percentil 99
    if 'TRIGLICERIDOS' in df.columns:
        df['TRIGLICERIDOS'] = np.clip(df['TRIGLICERIDOS'], None, df['TRIGLICERIDOS'].quantile(0.99))

    # Creatinina sérica — percentil 99
    if 'CREATININA_SERICA' in df.columns:
        df['CREATININA_SERICA'] = np.clip(df['CREATININA_SERICA'], None, df['CREATININA_SERICA'].quantile(0.99))

    return df


# ──────────────────────────────────────────────
# 3. ENCODING Y ESCALADO
# ──────────────────────────────────────────────

def codificar_y_escalar(df: pd.DataFrame, target: str = 'CLASIFICACION_RIESGO'):
    # One-hot encoding de categóricas (excepto target)
    categoricas = df.drop(columns=[target]).select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categoricas, drop_first=True)

    X = df.drop(columns=[target])
    y = df[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, X.columns.tolist(), le, scaler


# ──────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ──────────────────────────────────────────────

def recall_clase_alto(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[0], average='macro')


def entrenar(X_train, y_train, n_iter: int = 30):
    scorer_alto = make_scorer(recall_clase_alto)

    grid_rf = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=grid_rf,
        n_iter=n_iter,
        scoring=scorer_alto,
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    print("\n✔ Mejores hiperparámetros:")
    print(random_search.best_params_)
    print(f"\n✔ Mejor Recall clase ALTO: {random_search.best_score_:.4f}")

    return random_search.best_estimator_


# ──────────────────────────────────────────────
# 5. EVALUACIÓN
# ──────────────────────────────────────────────

def evaluar(modelo, X_test, y_test, le):
    y_pred = modelo.predict(X_test)
    print("\n══════════════════════════════")
    print("REPORTE DE CLASIFICACIÓN")
    print("══════════════════════════════")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nMATRIZ DE CONFUSIÓN")
    print(confusion_matrix(y_test, y_pred))


# ──────────────────────────────────────────────
# 6. GUARDADO DE ARTEFACTOS
# ──────────────────────────────────────────────

def guardar_artefactos(modelo, scaler, le, feature_names, carpeta: str = 'models'):
    os.makedirs(carpeta, exist_ok=True)
    joblib.dump(modelo, os.path.join(carpeta, 'modelo_rf.joblib'))
    joblib.dump(scaler, os.path.join(carpeta, 'scaler.joblib'))
    joblib.dump(le, os.path.join(carpeta, 'label_encoder.joblib'))
    with open(os.path.join(carpeta, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    print(f"\n✔ Artefactos guardados en '{carpeta}/'")


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento Random Forest - Riesgo Cardiovascular')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo Excel')
    parser.add_argument('--target', type=str, default='CLASIFICACION_RIESGO', help='Columna objetivo')
    parser.add_argument('--n_iter', type=int, default=30, help='Iteraciones del RandomizedSearchCV')
    parser.add_argument('--models_dir', type=str, default='models', help='Carpeta de salida para modelos')
    args = parser.parse_args()

    df = cargar_datos(args.data)
    df = preprocesar(df)
    X_scaled, y_encoded, feature_names, le, scaler = codificar_y_escalar(df, target=args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    modelo = entrenar(X_train, y_train, n_iter=args.n_iter)
    evaluar(modelo, X_test, y_test, le)
    guardar_artefactos(modelo, scaler, le, feature_names, carpeta=args.models_dir)


if __name__ == '__main__':
    main()
