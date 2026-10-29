"""Salvar e carregar dataset, resultados e modelos (seção 8 do notebook)."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import joblib
import pandas as pd

from classificador.avaliacao.metricas import matriz_confusao_df


def nome_arquivo_seguro(nome: str) -> str:
    """'Árvore de Decisão' -> 'arvore_de_decisao' (sem acento nem espaço)."""
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return sem_acento.lower().replace(" ", "_")


def salvar_dataset(df: pd.DataFrame, caminho: Path) -> Path:
    """Grava o dataset em CSV (cria a pasta se necessário)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8")
    return caminho


def salvar_modelo(modelo, features: list[str], classes: list[str], caminho: Path) -> Path:
    """Grava o modelo junto com as features e classes que ele espera."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "features": features, "classes": classes}, caminho)
    return caminho


def carregar_modelo(caminho: Path) -> dict:
    """Carrega um modelo salvo por :func:`salvar_modelo`."""
    return joblib.load(caminho)


def salvar_resultados(
    pasta: Path,
    df: pd.DataFrame,
    metricas: pd.DataFrame,
    modelos: dict,
    predicoes: dict,
    y_teste: pd.Series,
    features: list[str],
    classes: list[str],
) -> list[Path]:
    """Salva métricas, dataset usado, modelos e matrizes de confusão."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    metricas.to_csv(pasta / "comparacao_modelos.csv", index=False, encoding="utf-8-sig")
    df.to_csv(pasta / "dataset_utilizado.csv", index=False, encoding="utf-8-sig")

    for nome, modelo in modelos.items():
        base = nome_arquivo_seguro(nome)
        salvar_modelo(modelo, features, classes, pasta / f"{base}.joblib")
        matriz_confusao_df(y_teste, predicoes[nome], classes).to_csv(
            pasta / f"matriz_confusao_{base}.csv", encoding="utf-8-sig"
        )
    return sorted(p for p in pasta.iterdir() if p.is_file())
