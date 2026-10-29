"""Camada de dados: geração, validação e divisão do dataset."""

from classificador.dados.divisao import dividir_treino_teste
from classificador.dados.gerador import gerar_dataset
from classificador.dados.validacao import (
    contar_ausentes,
    contar_por_classe,
    selecionar_features,
    validar_dataset,
)

__all__ = [
    "contar_ausentes",
    "contar_por_classe",
    "dividir_treino_teste",
    "gerar_dataset",
    "selecionar_features",
    "validar_dataset",
]
