"""Processamento e extração de características de lotes de imagens reais.

Lê as imagens organizadas em pastas (por classe e partição), executa o
pipeline de Visão Computacional (OpenCV) e OCR (Pytesseract) e gera
o arquivo CSV de features (`features_passaportes.csv`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


from classificador.config import (
    COLUNA_ALVO,
    COLUNAS_DATASET,
    CSV_FEATURES,
)

EXTS_PADRAO = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

from classificador.extracao import extrair_features_de_imagem


def extrair_dataset_de_imagens(
    pasta_dataset: str | Path,
    classes: Sequence[str] = ("brasileiro", "estrangeiro", "invalido"),
    caminho_csv_saida: str | Path | None = CSV_FEATURES,
    extensoes: set[str] | None = None,
) -> pd.DataFrame:
    """Extrai características de todas as imagens em uma pasta de dataset.

    Suporta duas estruturas de diretórios:
    1. Estrutura particionada: `pasta/{train,val,test}/{classe}/*.jpg`
    2. Estrutura direta: `pasta/{classe}/*.jpg`

    Gera as colunas esperadas em :data:`COLUNAS_DATASET`:
    `nome_arquivo`, `caminho_arquivo`, `x_face`, `y_face`, `w_face`, `h_face`,
    `quantidade_palavras`, `presenca_qrcode`, `presenca_rostos`, contadores de termos,
    `label` e `texto_ocr`.
    """
    pasta = Path(pasta_dataset)
    if not pasta.exists():
        raise FileNotFoundError(f"Pasta do dataset não encontrada: {pasta}")

    exts = extensoes or {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    linhas = []

    # Identifica se há partições (train, val, test) ou classes diretas
    subpastas = [p for p in pasta.iterdir() if p.is_dir()]
    tem_particoes = any(p.name in ("train", "val", "test") for p in subpastas)

    tarefas = []
    if tem_particoes:
        for part in ("train", "val", "test"):
            pasta_part = pasta / part
            if not pasta_part.exists():
                continue
            for classe in classes:
                pasta_classe = pasta_part / classe
                if not pasta_classe.exists():
                    continue
                for arq in sorted(pasta_classe.iterdir()):
                    if arq.is_file() and arq.suffix.lower() in exts:
                        tarefas.append((arq, classe, part))
    else:
        for classe in classes:
            pasta_classe = pasta / classe
            if not pasta_classe.exists():
                continue
            for arq in sorted(pasta_classe.iterdir()):
                if arq.is_file() and arq.suffix.lower() in exts:
                    tarefas.append((arq, classe, None))

    if not tarefas:
        raise ValueError(f"Nenhuma imagem encontrada nas classes {classes} em {pasta}")

    print(f"Extraindo características de {len(tarefas)} imagens reais com OpenCV + OCR...")

    for caminho_img, classe, particao in tqdm(tarefas, desc="Extração de imagens"):
        try:
            df_feat, texto = extrair_features_de_imagem(caminho_img)
            linha = df_feat.iloc[0].to_dict()
            linha["nome_arquivo"] = caminho_img.name
            linha["caminho_arquivo"] = caminho_img.relative_to(pasta.parent).as_posix() if pasta.parent in caminho_img.parents else caminho_img.as_posix()
            linha[COLUNA_ALVO] = classe
            linha["texto_ocr"] = texto
            if particao:
                linha["particao"] = particao
            linhas.append(linha)
        except Exception as e:
            print(f"Aviso: falha ao extrair features de {caminho_img.name}: {e}")

    df_resultado = pd.DataFrame(linhas)

    # Garante que todas as colunas de COLUNAS_DATASET existam
    for col in COLUNAS_DATASET:
        if col not in df_resultado.columns:
            df_resultado[col] = 0

    if caminho_csv_saida:
        caminho_csv = Path(caminho_csv_saida)
        caminho_csv.parent.mkdir(parents=True, exist_ok=True)
        df_resultado.to_csv(caminho_csv, index=False)
        print(f"Features extraídas salvas em: {caminho_csv} ({len(df_resultado)} registros)")

    return df_resultado
