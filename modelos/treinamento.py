"""Treinamento dos classificadores."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


def treinar_modelos(
    modelos: dict[str, Pipeline],
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    verbose: bool = True,
) -> dict[str, Pipeline]:
    """Ajusta (in-place) cada modelo com os dados de treino e os devolve."""
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        if verbose:
            print(f"Treinado com sucesso: {nome}")
    return modelos
