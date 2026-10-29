import pandas as pd
import pytest

from classificador.config import COLUNAS_DATASET
from classificador.dados import (
    contar_ausentes,
    dividir_treino_teste,
    gerar_dataset,
    selecionar_features,
    validar_dataset,
)
from classificador.extracao import FEATURES_EXTRATOR


@pytest.fixture(scope="module")
def df():
    return gerar_dataset()


def test_formato_do_dataset(df):
    assert df.shape == (300, 25)
    assert list(df.columns) == COLUNAS_DATASET
    assert df["label"].value_counts().to_dict() == {
        "brasileiro": 100, "estrangeiro": 100, "invalido": 100
    }


def test_dataset_e_reprodutivel(df):
    pd.testing.assert_frame_equal(df, gerar_dataset())


def test_reproduz_os_ausentes_do_colab(df):
    """Contagens idênticas às impressas no notebook original."""
    ausentes = contar_ausentes(df)["ausentes"].to_dict()
    assert ausentes == {
        "x_face": 37, "y_face": 36, "w_face": 37, "h_face": 27, "quantidade_palavras": 34
    }


def test_features_sao_as_21_do_extrator(df):
    features = selecionar_features(df)
    assert len(features) == 21
    assert features == FEATURES_EXTRATOR
    for proibida in ("nome_arquivo", "caminho_arquivo", "texto_ocr", "label"):
        assert proibida not in features


def test_validacao_rejeita_dataset_incompleto(df):
    with pytest.raises(ValueError, match="linhas"):
        validar_dataset(df.head(10), selecionar_features(df))
    with pytest.raises(ValueError, match="característica"):
        validar_dataset(df, [])


def test_divisao_estratificada(df):
    X_tr, X_te, y_tr, y_te = dividir_treino_teste(df, selecionar_features(df))
    assert X_tr.shape == (225, 21) and X_te.shape == (75, 21)
    assert y_te.value_counts().to_dict() == {"brasileiro": 25, "estrangeiro": 25, "invalido": 25}
    assert set(X_tr.index).isdisjoint(X_te.index)
