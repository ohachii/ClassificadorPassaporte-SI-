import shutil

import pytest

from classificador.extracao import FEATURES_EXTRATOR
from classificador.extracao.imagem import contar_termos
from classificador.modelos import criar_modelos
from classificador.persistencia import carregar_modelo, nome_arquivo_seguro, salvar_modelo


def test_nome_arquivo_seguro():
    assert nome_arquivo_seguro("Árvore de Decisão") == "arvore_de_decisao"
    assert nome_arquivo_seguro("Random Forest") == "random_forest"
    assert nome_arquivo_seguro("KNN") == "knn"


def test_salvar_e_carregar_modelo(tmp_path):
    modelo = criar_modelos()["KNN"]
    caminho = salvar_modelo(modelo, ["a", "b"], ["x", "y"], tmp_path / "pasta" / "m.joblib")
    dados = carregar_modelo(caminho)
    assert set(dados) == {"modelo", "features", "classes"}
    assert dados["features"] == ["a", "b"] and dados["classes"] == ["x", "y"]


def test_contagem_de_termos():
    termos = contar_termos("Carteira de ESTUDANTE, matrícula 123. Passaporte / passport. CPF")
    assert termos["qtd_estudante"] == 2.0   # "estudante" + "carteira"
    assert termos["qtd_matricula"] == 1.0
    assert termos["qtd_passaporte"] == 2.0  # "passaporte" + "passport"
    assert termos["qtd_cpf"] == 1.0
    assert termos["qtd_cnh"] == 0.0
    assert list(termos) == FEATURES_EXTRATOR[7:]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract não instalado")
def test_extracao_de_imagem_real(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("pytesseract")
    import numpy as np

    from classificador.extracao import imagem

    monkeypatch.setattr(imagem, "IDIOMA_OCR", "eng")  # ASCII puro: dispensa o pacote 'por'
    img = np.full((300, 800, 3), 255, np.uint8)
    cv2.putText(img, "CARTEIRA DE ESTUDANTE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    pasta = tmp_path / "Área de Trabalho"  # caminho com acento (caso típico no Windows)
    pasta.mkdir()
    cv2.imwrite(str(pasta / "doc.png"), img)

    features, texto = imagem.extrair_features_de_imagem(pasta / "doc.png")
    assert list(features.columns) == FEATURES_EXTRATOR
    assert features.shape == (1, 21)
    assert features.loc[0, "qtd_estudante"] == 2.0
    assert features.loc[0, "presenca_rostos"] == 0.0


def test_imagem_inexistente_gera_erro_claro(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("pytesseract")
    from classificador.extracao import extrair_features_de_imagem

    with pytest.raises(FileNotFoundError):
        extrair_features_de_imagem(tmp_path / "nao_existe.png")
