"""Camada de avaliação: métricas, gráficos e seleção do melhor modelo."""

from classificador.avaliacao.graficos import plotar_comparacao, plotar_matriz_confusao
from classificador.avaliacao.metricas import (
    avaliar_modelos,
    matriz_confusao_df,
    relatorio_classificacao,
)
from classificador.avaliacao.selecao import ResultadoSelecao, escolher_melhor_modelo

__all__ = [
    "ResultadoSelecao",
    "avaliar_modelos",
    "escolher_melhor_modelo",
    "matriz_confusao_df",
    "plotar_comparacao",
    "plotar_matriz_confusao",
    "relatorio_classificacao",
]
