"""Geração do dataset sintético (seção 2 do notebook).

O dataset simula as características extraídas de imagens de documentos
(posição do rosto, quantidade de palavras, contagem de termos do OCR etc.)
para três classes: ``brasileiro``, ``estrangeiro`` e ``invalido``.

IMPORTANTE: a ORDEM das chamadas ao gerador aleatório foi mantida idêntica à do
notebook original. Com ``SEED = 42`` o dataset resultante é exatamente o mesmo
que foi usado no Colab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from classificador.config import (
    COLUNAS_COM_NULOS,
    COLUNAS_DATASET,
    EXEMPLOS_POR_CLASSE,
    PROPORCAO_NULOS,
    SEED,
)


def _linha(nome: str, pasta: str, label: str, texto_ocr: str, **valores) -> dict:
    """Monta uma linha completa; toda coluna não informada vale 0."""
    linha = {coluna: 0 for coluna in COLUNAS_DATASET}
    linha.update(
        nome_arquivo=nome,
        caminho_arquivo=f"dataset/{pasta}/{nome}",
        label=label,
        texto_ocr=texto_ocr,
    )
    linha.update(valores)
    return linha


def _gerar_brasileiros(rng: np.random.RandomState, n: int) -> list[dict]:
    """Brasileiros: rosto presente, alta variação e ruído de OCR."""
    registros = []
    for i in range(1, n + 1):
        x = rng.randint(10, 50)
        y = rng.randint(15, 60)
        w = rng.randint(60, 140)
        h = rng.randint(60, 140)
        qtd_palavras = rng.randint(12, 26)

        # Ruído de OCR: às vezes o termo de passaporte não é lido.
        qtd_passaporte = rng.choice([1, 0], p=[0.70, 0.30])
        qtd_nacional = rng.choice([1, 0], p=[0.60, 0.40])
        qtd_validade = rng.choice([1, 0], p=[0.80, 0.20])

        # Pequena chance de termos de outros documentos por sujeira no OCR.
        qtd_rg = rng.choice([1, 0], p=[0.15, 0.85])
        qtd_estudante = rng.choice([1, 0], p=[0.05, 0.95])

        registros.append(_linha(
            f"brasileiro_{i:03d}.png", "brasileiro", "brasileiro", "PASSPORT",
            x_face=x, y_face=y, w_face=w, h_face=h,
            quantidade_palavras=qtd_palavras,
            presenca_rostos=1,
            qtd_rg=qtd_rg,
            qtd_passaporte=qtd_passaporte,
            qtd_validade=qtd_validade,
            qtd_nacional=qtd_nacional,
            qtd_estudante=qtd_estudante,
        ))
    return registros


def _gerar_estrangeiros(rng: np.random.RandomState, n: int) -> list[dict]:
    """Estrangeiros: podem ou não ter rosto detectado; podem ter QR Code."""
    registros = []
    for i in range(1, n + 1):
        tem_rosto = rng.choice([0, 1], p=[0.30, 0.70])
        if tem_rosto:
            x, y, w, h = (
                rng.randint(15, 65), rng.randint(20, 70),
                rng.randint(50, 130), rng.randint(50, 130),
            )
        else:
            x, y, w, h = 0, 0, 0, 0
        qtd_palavras = rng.randint(10, 24)

        # O termo genérico "passport" também aparece em estrangeiros.
        qtd_passaporte = rng.choice([1, 0], p=[0.45, 0.55])
        qtd_validade = rng.choice([1, 0], p=[0.10, 0.90])
        qrcode = rng.choice([1, 0], p=[0.35, 0.65])

        registros.append(_linha(
            f"estrangeiro_{i:03d}.png", "estrangeiro", "estrangeiro",
            "FOREIGN_PASSPORT",
            x_face=x, y_face=y, w_face=w, h_face=h,
            quantidade_palavras=qtd_palavras,
            presenca_qrcode=qrcode,
            presenca_rostos=tem_rosto,
            qtd_passaporte=qtd_passaporte,
            qtd_validade=qtd_validade,
        ))
    return registros


def _gerar_invalidos(rng: np.random.RandomState, n: int) -> list[dict]:
    """Inválidos (ex.: carteiras estudantis) com semelhança e ruído de OCR."""
    registros = []
    for i in range(1, n + 1):
        tem_rosto = rng.choice([0, 1], p=[0.40, 0.60])
        if tem_rosto:
            x, y, w, h = (
                rng.randint(5, 35), rng.randint(5, 35),
                rng.randint(30, 90), rng.randint(30, 90),
            )
        else:
            x, y, w, h = 0, 0, 0, 0
        qtd_palavras = rng.randint(15, 38)

        # O OCR pode ler "passaporte"/"validade" por engano.
        qtd_passaporte = rng.choice([1, 0], p=[0.15, 0.85])
        qtd_validade = rng.choice([1, 0], p=[0.20, 0.80])
        qtd_estudante = rng.choice([1, 2, 0], p=[0.60, 0.20, 0.20])
        qtd_matricula = rng.choice([1, 0], p=[0.50, 0.50])

        registros.append(_linha(
            f"invalido_{i:03d}.png", "invalido", "invalido", "STUDENT_CARD",
            x_face=x, y_face=y, w_face=w, h_face=h,
            quantidade_palavras=qtd_palavras,
            presenca_rostos=tem_rosto,
            qtd_documento=1,
            qtd_passaporte=qtd_passaporte,
            qtd_validade=qtd_validade,
            qtd_estudante=qtd_estudante,
            qtd_matricula=qtd_matricula,
        ))
    return registros


def _inserir_nulos(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Insere NaN aleatórios para que o Imputer tenha o que tratar."""
    for coluna in COLUNAS_COM_NULOS:
        df[coluna] = df[coluna].astype(float)
        mascara = rng.rand(len(df)) < PROPORCAO_NULOS
        df.loc[mascara, coluna] = np.nan
    return df


def gerar_dataset(seed: int = SEED, por_classe: int = EXEMPLOS_POR_CLASSE) -> pd.DataFrame:
    """Gera o dataset sintético completo (3 classes x ``por_classe`` linhas)."""
    rng = np.random.RandomState(seed)

    registros = (
        _gerar_brasileiros(rng, por_classe)
        + _gerar_estrangeiros(rng, por_classe)
        + _gerar_invalidos(rng, por_classe)
    )
    df = pd.DataFrame(registros, columns=COLUNAS_DATASET)
    return _inserir_nulos(df, rng)
