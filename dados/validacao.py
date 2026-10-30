"""Validação e entendimento dos dados (seção 3 do notebook)."""

from __future__ import annotations

import pandas as pd

from classificador.config import (
    CLASSES_ESPERADAS,
    COLUNA_ALVO,
    COLUNAS_NAO_TREINAVEIS,
    LINHAS_ESPERADAS,
)


def selecionar_features(df: pd.DataFrame) -> list[str]:
    """Colunas numéricas usadas no treino (exclui identificadores, OCR e alvo)."""
    excluidas = {*COLUNAS_NAO_TREINAVEIS, COLUNA_ALVO}
    return [
        coluna for coluna in df.columns
        if coluna not in excluidas and pd.api.types.is_numeric_dtype(df[coluna])
    ]


def contar_por_classe(df: pd.DataFrame) -> pd.DataFrame:
    """Quantidade de exemplos por classe."""
    return df[COLUNA_ALVO].value_counts().rename_axis("classe").reset_index(name="quantidade")


def contar_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas com valores ausentes (que o Imputer resolve no pipeline)."""
    return df.isna().sum().to_frame("ausentes").query("ausentes > 0")


def validar_dataset(
    df: pd.DataFrame,
    features: list[str],
    linhas_esperadas: int | None = LINHAS_ESPERADAS,
) -> None:
    """Interrompe com ``ValueError`` se o dataset não estiver como esperado."""
    if len(df) == 0:
        raise ValueError("O dataset está vazio.")
    if linhas_esperadas is not None and len(df) != linhas_esperadas:
        raise ValueError(f"Esperadas {linhas_esperadas} linhas; encontradas {len(df)}.")
    if df[COLUNA_ALVO].nunique() != CLASSES_ESPERADAS:
        raise ValueError(f"Esperadas exatamente {CLASSES_ESPERADAS} classes.")
    if not features:
        raise ValueError("Nenhuma característica numérica encontrada.")

