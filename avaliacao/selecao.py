"""Escolha do melhor modelo.

Critério (documentado e reprodutível):

1. Ordena-se pelo **F1 macro no conjunto de teste** (o mesmo critério que o
   notebook usa para ordenar a tabela de métricas).
2. Com apenas 75 exemplos de teste, UM exemplo errado já muda a acurácia em
   ~1,3 ponto percentual. Portanto, modelos cujo F1 macro esteja a menos de
   *um exemplo de teste* do primeiro colocado são considerados **tecnicamente
   empatados**.
3. Os empatados são desempatados pela **média do F1 macro em validação cruzada
   estratificada repetida, calculada SOMENTE com os dados de treino**. O
   conjunto de teste não é usado nesse desempate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from classificador.config import CV_FOLDS, CV_REPETICOES, SEED


@dataclass(frozen=True)
class ResultadoSelecao:
    """Resultado da escolha do melhor modelo."""

    melhor: str
    finalistas: pd.DataFrame  # tabela com os modelos tecnicamente empatados
    tolerancia: float
    justificativa: str


def escolher_melhor_modelo(
    metricas: pd.DataFrame,
    modelos: dict[str, Pipeline],
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    n_teste: int,
    seed: int = SEED,
) -> ResultadoSelecao:
    """Escolhe o melhor modelo conforme o critério descrito no módulo."""
    tolerancia = 1.0 / n_teste
    topo = metricas["f1_macro"].max()
    tabela = metricas[metricas["f1_macro"] >= topo - tolerancia].copy()
    tabela = tabela[["modelo", "f1_macro"]].rename(columns={"f1_macro": "f1_macro_teste"})

    if len(tabela) == 1:
        melhor = tabela.iloc[0]["modelo"]
        return ResultadoSelecao(
            melhor=melhor,
            finalistas=tabela.reset_index(drop=True),
            tolerancia=tolerancia,
            justificativa=f"{melhor} teve o maior F1 macro no teste, sem empate técnico.",
        )

    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPETICOES, random_state=seed)
    medias, desvios = [], []
    for nome in tabela["modelo"]:
        scores = cross_validate(modelos[nome], X_treino, y_treino, cv=cv, scoring="f1_macro")
        medias.append(scores["test_score"].mean())
        desvios.append(scores["test_score"].std())
    tabela["f1_macro_cv_media"] = medias
    tabela["f1_macro_cv_desvio"] = desvios
    tabela = tabela.sort_values("f1_macro_cv_media", ascending=False).reset_index(drop=True)

    melhor = tabela.iloc[0]["modelo"]
    nomes = ", ".join(tabela["modelo"])
    justificativa = (
        f"Empate técnico no teste entre: {nomes} (diferença de F1 macro menor que "
        f"1 exemplo de teste = {tolerancia:.4f}). Desempate por validação cruzada "
        f"{CV_FOLDS}-fold x {CV_REPETICOES} repetições nos dados de treino: "
        f"{melhor} obteve a maior média de F1 macro "
        f"({tabela.iloc[0]['f1_macro_cv_media']:.4f})."
    )
    return ResultadoSelecao(melhor, tabela, tolerancia, justificativa)
