"""Camada de modelos: fábrica dos classificadores e treinamento."""

from classificador.modelos.fabrica import criar_modelos
from classificador.modelos.treinamento import treinar_modelos

__all__ = ["criar_modelos", "treinar_modelos"]
