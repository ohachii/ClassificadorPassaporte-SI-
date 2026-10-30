"""Camada de dados: geração, deduplicação, validação e divisão do dataset."""

from classificador.dados.deduplicacao import (
    auditar_unicidade_dataset,
    calcular_dhash,
    extrair_id_base,
    filtrar_identidades_unicas,
)
from classificador.dados.divisao import dividir_treino_teste
from classificador.dados.extracao_dataset import extrair_dataset_de_imagens
from classificador.dados.gerador import carregar_ou_gerar_dataset, gerar_dataset
from classificador.dados.validacao import (
    contar_ausentes,
    contar_por_classe,
    selecionar_features,
    validar_dataset,
)

__all__ = [
    "auditar_unicidade_dataset",
    "calcular_dhash",
    "carregar_ou_gerar_dataset",
    "contar_ausentes",
    "contar_por_classe",
    "dividir_treino_teste",
    "extrair_dataset_de_imagens",
    "extrair_id_base",
    "filtrar_identidades_unicas",
    "gerar_dataset",
    "selecionar_features",
    "validar_dataset",
]

