"""Separação treino/teste (seção 4 do notebook)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from classificador.config import COLUNA_ALVO, SEED, TAMANHO_TESTE


def dividir_treino_teste(
    df: pd.DataFrame,
    features: list[str],
    tamanho_teste: float = TAMANHO_TESTE,
    seed: int = SEED,
):
    """Divisão estratificada: mantém a proporção das três classes.

    Retorna ``X_treino, X_teste, y_treino, y_teste``.
    """
    X = df[features].copy()
    y = df[COLUNA_ALVO].astype(str)
    return train_test_split(
        X, y, test_size=tamanho_teste, random_state=seed, stratify=y
    )
