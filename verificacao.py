"""Verificação de compatibilidade extrator <-> modelo (seção 10 do notebook).

Garante que as colunas geradas pelo extrator de imagens são exatamente as que o
modelo salvo espera (mesmos nomes, mesma ordem) e que o pipeline consegue
prever com elas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from classificador.extracao.imagem import FEATURES_EXTRATOR
from classificador.persistencia import carregar_modelo


def verificar_compatibilidade(caminho_modelo: Path, verbose: bool = True) -> list[str]:
    """Retorna a lista de problemas encontrados (vazia = tudo compatível)."""
    erros: list[str] = []
    log = print if verbose else (lambda *a, **k: None)

    try:
        dados = carregar_modelo(caminho_modelo)
    except Exception as erro:  # noqa: BLE001 - qualquer falha de leitura é um problema
        return [f"Falha ao carregar modelo salvo: {erro}"]

    features_modelo = dados["features"]
    log(f"OK: modelo carregado ({len(features_modelo)} características esperadas).")

    if features_modelo != FEATURES_EXTRATOR:
        erros.append(
            "Divergência de características!\n"
            f"  Esperadas pelo modelo: {features_modelo}\n"
            f"  Geradas pelo extrator: {FEATURES_EXTRATOR}"
        )
    else:
        log("OK: o extrator gera as mesmas variáveis, na ordem correta.")

    zeros = pd.DataFrame([np.zeros(len(FEATURES_EXTRATOR))], columns=FEATURES_EXTRATOR)
    try:
        classe = dados["modelo"].predict(zeros)[0]
        log(f"OK: teste de sanidade do pipeline aprovado (classe simulada: '{classe}').")
    except Exception as erro:  # noqa: BLE001
        erros.append(f"Erro ao processar as colunas no pipeline do classificador: {erro}")

    if verbose:
        if erros:
            print("\nRISCO DE FALHA DETECTADO:")
            for erro in erros:
                print("-", erro)
        else:
            print("\nOK: o pipeline de imagem real é compatível com o modelo treinado.")
    return erros
