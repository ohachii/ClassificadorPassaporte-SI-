"""Gráficos de avaliação (seções 6 e 7 do notebook).

Cada função pode salvar a figura (``salvar_em``) e/ou exibi-la (``mostrar``).
Em scripts, deixe ``mostrar=False`` para não abrir janelas; em notebooks, use
``mostrar=True``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _finalizar(fig, salvar_em: Path | None, mostrar: bool) -> None:
    fig.tight_layout()
    if salvar_em is not None:
        Path(salvar_em).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=120, bbox_inches="tight")
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


def plotar_matriz_confusao(
    y_teste,
    pred,
    classes: list[str],
    nome_modelo: str,
    salvar_em: Path | None = None,
    mostrar: bool = False,
) -> None:
    """Matriz de confusão de um modelo."""
    cm = confusion_matrix(y_teste, pred, labels=classes)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, values_format="d", xticks_rotation=30
    )
    ax.set_title(f"Matriz de confusão — {nome_modelo}")
    _finalizar(fig, salvar_em, mostrar)


def plotar_comparacao(
    metricas,
    salvar_em: Path | None = None,
    mostrar: bool = False,
) -> None:
    """Barras comparando acurácia, precisão, recall e F1 dos modelos."""
    colunas = ["acuracia", "precisao_macro", "recall_macro", "f1_macro"]
    fig, ax = plt.subplots(figsize=(10, 5))
    metricas.set_index("modelo")[colunas].plot(kind="bar", ax=ax, ylim=(0, 1.05))
    ax.set_title("Comparação dos classificadores (Dataset de Alta Variabilidade)")
    ax.set_ylabel("Pontuação")
    ax.set_xlabel("Modelo")
    plt.setp(ax.get_xticklabels(), rotation=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))  # fora da área das barras
    _finalizar(fig, salvar_em, mostrar)
