"""Métricas de avaliação (seção 6 do notebook)."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def avaliar_modelos(
    modelos: dict[str, Pipeline],
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Avalia todos os modelos no conjunto de teste.

    Retorna ``(metricas, predicoes)``: a tabela comparativa (ordenada por
    F1 macro, do melhor para o pior) e as predições de cada modelo.
    """
    resultados, predicoes = [], {}
    for nome, modelo in modelos.items():
        pred = modelo.predict(X_teste)
        predicoes[nome] = pred
        resultados.append({
            "modelo": nome,
            "acuracia": accuracy_score(y_teste, pred),
            "precisao_macro": precision_score(y_teste, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_teste, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_teste, pred, average="macro", zero_division=0),
        })
    metricas = (
        pd.DataFrame(resultados)
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    return metricas, predicoes


def relatorio_classificacao(y_teste, pred) -> str:
    """Relatório por classe (precisão, recall, F1 e suporte)."""
    return classification_report(y_teste, pred, zero_division=0)


def matriz_confusao_df(y_teste, pred, classes: list[str]) -> pd.DataFrame:
    """Matriz de confusão como DataFrame (linhas = real, colunas = previsto)."""
    return pd.DataFrame(
        confusion_matrix(y_teste, pred, labels=classes),
        index=classes,
        columns=classes,
    )
