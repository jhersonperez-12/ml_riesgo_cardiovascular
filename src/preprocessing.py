"""
preprocessing.py — Funciones de preprocesamiento compartidas entre train.py y app.py
"""

import numpy as np
import pandas as pd


VALORES_INVALIDOS = [99999, 999, 9999, 99]
COLUMNAS_IQR = ['PESO', 'TALLA', 'EDAD', 'IMC']


def limpiar_invalidos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace(VALORES_INVALIDOS, np.nan, inplace=True)
    return df


def imputar_mediana(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df


def capar_iqr(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    Q1, Q3 = df[columna].quantile(0.25), df[columna].quantile(0.75)
    IQR = Q3 - Q1
    df[columna] = np.clip(df[columna], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df


def tratar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNAS_IQR:
        if col in df.columns:
            df = capar_iqr(df, col)

    if 'COLESTEROL_HDL' in df.columns:
        df['COLESTEROL_HDL'] = np.clip(df['COLESTEROL_HDL'], 20, None)

    if 'COLESTEROL_LDL' in df.columns:
        mediana_ldl = df[df['COLESTEROL_LDL'] >= 20]['COLESTEROL_LDL'].median()
        df.loc[df['COLESTEROL_LDL'] < 20, 'COLESTEROL_LDL'] = mediana_ldl

    if 'TRIGLICERIDOS' in df.columns:
        df['TRIGLICERIDOS'] = np.clip(df['TRIGLICERIDOS'], None, df['TRIGLICERIDOS'].quantile(0.99))

    if 'CREATININA_SERICA' in df.columns:
        df['CREATININA_SERICA'] = np.clip(df['CREATININA_SERICA'], None, df['CREATININA_SERICA'].quantile(0.99))

    return df


def pipeline_completo(df: pd.DataFrame) -> pd.DataFrame:
    df = limpiar_invalidos(df)
    df = imputar_mediana(df)
    df = tratar_outliers(df)
    return df
