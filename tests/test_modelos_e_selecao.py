import pandas as pd
import pytest

from classificador.avaliacao import avaliar_modelos, escolher_melhor_modelo
from classificador.dados import dividir_treino_teste, gerar_dataset, selecionar_features
from classificador.modelos import criar_modelos, treinar_modelos

# Acurácias obtidas no notebook do Colab (tolerância de 3 exemplos de teste).
ACURACIA_COLAB = {"KNN": 0.92, "Árvore de Decisão": 0.96, "Random Forest": 0.96}


@pytest.fixture(scope="module")
def treinado():
    df = gerar_dataset()
    X_tr, X_te, y_tr, y_te = dividir_treino_teste(df, selecionar_features(df))
    modelos = treinar_modelos(criar_modelos(), X_tr, y_tr, verbose=False)
    metricas, predicoes = avaliar_modelos(modelos, X_te, y_te)
    return modelos, metricas, predicoes, X_tr, y_tr, X_te, y_te


def test_tres_modelos_criados():
    assert list(criar_modelos()) == ["KNN", "Árvore de Decisão", "Random Forest"]


def test_metricas_proximas_das_do_colab(treinado):
    _, metricas, *_ = treinado
    obtido = metricas.set_index("modelo")["acuracia"].to_dict()
    for nome, esperado in ACURACIA_COLAB.items():
        assert obtido[nome] == pytest.approx(esperado, abs=0.04), nome
    assert list(metricas.columns) == [
        "modelo", "acuracia", "precisao_macro", "recall_macro", "f1_macro"
    ]


def test_modelos_toleram_valores_ausentes(treinado):
    modelos, *_, X_te, _ = treinado
    com_nan = X_te.copy()
    com_nan.iloc[:, :5] = float("nan")
    for modelo in modelos.values():
        assert len(modelo.predict(com_nan)) == len(X_te)


def test_selecao_desempata_por_validacao_cruzada(treinado):
    modelos, metricas, _, X_tr, y_tr, X_te, _ = treinado
    r = escolher_melhor_modelo(metricas, modelos, X_tr, y_tr, n_teste=len(X_te))
    assert r.melhor == "Random Forest"
    assert set(r.finalistas["modelo"]) == {"Random Forest", "Árvore de Decisão"}
    assert "f1_macro_cv_media" in r.finalistas.columns
    assert "KNN" not in set(r.finalistas["modelo"])  # claramente pior: fora do empate


def test_selecao_sem_empate_nao_usa_validacao_cruzada(treinado):
    modelos, _, _, X_tr, y_tr, X_te, _ = treinado
    metricas = pd.DataFrame({
        "modelo": ["KNN", "Random Forest"],
        "acuracia": [0.99, 0.90], "precisao_macro": [0.99, 0.90],
        "recall_macro": [0.99, 0.90], "f1_macro": [0.99, 0.90],
    })
    r = escolher_melhor_modelo(metricas, modelos, X_tr, y_tr, n_teste=len(X_te))
    assert r.melhor == "KNN"
    assert "f1_macro_cv_media" not in r.finalistas.columns
