"""Exportação do melhor modelo para a pasta ``melhor_modelo/``.

A pasta gerada é autossuficiente: contém o modelo, o extrator de imagens, a
lista de dependências (com a versão exata do scikit-learn usada no treino),
metadados e exemplos de teste. Os arquivos ``predict.py`` e ``README.md`` dessa
pasta são escritos à mão e NÃO são sobrescritos aqui.
"""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from classificador.avaliacao.selecao import ResultadoSelecao
from classificador.config import COLUNA_ALVO, DIR_MELHOR_MODELO, SEED
from classificador.persistencia import salvar_modelo

AVISO = (
    "Aviso acadêmico: protótipo desenvolvido para a disciplina de Sistemas Inteligentes "
    "com garantia de fotos e pessoas únicas. Não use para decisões oficiais."
)


_EXTRATOR_ORIGEM = Path(__file__).resolve().parent / "extracao" / "imagem.py"


def _versao(pacote: str) -> str:
    return metadata.version(pacote)


def _escrever_requirements(destino: Path) -> Path:
    texto = (
        "# Dependências para executar o melhor modelo.\n"
        "# Arquivo GERADO por `python main.py treinar`; não edite à mão.\n"
        "#\n"
        "# O scikit-learn está FIXADO na versão usada no treino: o arquivo\n"
        "# modelo.joblib só é garantido nessa versão.\n"
        f"scikit-learn=={_versao('scikit-learn')}\n"
        "numpy\n"
        "pandas\n"
        "joblib\n"
        "\n"
        "# Necessários apenas para classificar IMAGENS (exige também o programa\n"
        "# Tesseract instalado no sistema; veja o README.md desta pasta).\n"
        "opencv-python-headless\n"
        "pytesseract\n"
    )
    caminho = destino / "requirements.txt"
    caminho.write_text(texto, encoding="utf-8")
    return caminho


def _escrever_exemplos(destino: Path, X_teste: pd.DataFrame, y_teste: pd.Series) -> Path:
    """Até 3 exemplos por classe, retirados do conjunto de TESTE (não vistos no treino)."""
    amostra = X_teste.assign(**{"label_real": y_teste})
    amostra = amostra.groupby("label_real", sort=True).head(3).sort_values("label_real")
    pasta = destino / "exemplos"
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / "exemplos_teste.csv"
    amostra.to_csv(caminho, index=False, encoding="utf-8")
    return caminho


def exportar_melhor_modelo(
    nome: str,
    modelo,
    features: list[str],
    classes: list[str],
    metricas_modelo: dict,
    selecao: ResultadoSelecao,
    X_treino: pd.DataFrame,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    destino: Path = DIR_MELHOR_MODELO,
) -> list[Path]:
    """Grava modelo, metadados, dependências, extrator e exemplos em ``destino``."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    arquivos = [salvar_modelo(modelo, features, classes, destino / "modelo.joblib")]

    info = {
        "modelo": nome,
        "classes": classes,
        "features": features,
        "metricas_teste": {
            **{k: round(float(v), 4) for k, v in metricas_modelo.items() if k != "modelo"},
            "n_teste": int(len(X_teste)),
        },
        "selecao": {
            "justificativa": selecao.justificativa,
            "tolerancia_empate": round(selecao.tolerancia, 4),
            "finalistas": json.loads(selecao.finalistas.round(4).to_json(orient="records")),
        },
        "treino": {
            "n_treino": int(len(X_treino)),
            "seed": SEED,
            "alvo": COLUNA_ALVO,
            "dados": "sintéticos (gerados por classificador/dados/gerador.py)",
        },
        "versoes": {
            "python": platform.python_version(),
            "scikit-learn": _versao("scikit-learn"),
            "numpy": _versao("numpy"),
            "pandas": _versao("pandas"),
            "joblib": _versao("joblib"),
        },
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "aviso": AVISO,
    }
    caminho_meta = destino / "metadata.json"
    caminho_meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    arquivos.append(caminho_meta)

    arquivos.append(_escrever_requirements(destino))
    arquivos.append(_escrever_exemplos(destino, X_teste, y_teste))

    # Cópia do extrator (módulo autocontido) para a pasta rodar sozinha.
    extrator = destino / "extrator_imagem.py"
    shutil.copyfile(_EXTRATOR_ORIGEM, extrator)
    arquivos.append(extrator)
    return arquivos
