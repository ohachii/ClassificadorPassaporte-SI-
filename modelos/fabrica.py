"""Fábrica dos classificadores (seção 5 do notebook).

Padrão Factory: cada algoritmo tem sua função de criação, registrada em
``_FABRICAS``. Para incluir um novo modelo basta escrever uma função e
registrá-la; nenhum outro módulo precisa mudar.

Todos os modelos são ``Pipeline`` do scikit-learn: o Imputer (e o scaler, quando
necessário) fica DENTRO do pipeline, então é ajustado só com dados de treino.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from classificador.config import SEED


def _criar_knn(seed: int) -> Pipeline:
    # KNN é baseado em distância, por isso precisa de padronização.
    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=5)),
    ])


def _criar_arvore(seed: int) -> Pipeline:
    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("modelo", DecisionTreeClassifier(
            max_depth=5, random_state=seed, class_weight="balanced"
        )),
    ])


def _criar_random_forest(seed: int) -> Pipeline:
    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=200, random_state=seed,
            class_weight="balanced", max_features="sqrt",
        )),
    ])


_FABRICAS: dict[str, Callable[[int], Pipeline]] = {
    "KNN": _criar_knn,
    "Árvore de Decisão": _criar_arvore,
    "Random Forest": _criar_random_forest,
}


def criar_modelos(seed: int = SEED) -> dict[str, Pipeline]:
    """Cria os três classificadores (ainda não treinados)."""
    return {nome: fabrica(seed) for nome, fabrica in _FABRICAS.items()}
