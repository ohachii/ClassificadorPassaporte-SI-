"""Teste de integração: roda o experimento todo em pasta temporária e executa
a pasta ``melhor_modelo`` gerada, isolada do restante do projeto."""

import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from classificador.experimento import executar_experimento


@pytest.fixture(scope="module")
def projeto(tmp_path_factory):
    base = tmp_path_factory.mktemp("saida")
    melhor = base / "melhor_modelo"
    # predict.py e README são arquivos escritos à mão: copiamos o original, como no repositório.
    from classificador.config import DIR_MELHOR_MODELO

    melhor.mkdir()
    shutil.copyfile(DIR_MELHOR_MODELO / "predict.py", melhor / "predict.py")

    resultado = executar_experimento(
        salvar=True,
        exportar=True,
        mostrar_graficos=False,
        pasta_resultados=base / "resultados",
        pasta_melhor_modelo=melhor,
        arquivo_csv=base / "data" / "features_passaportes.csv",
    )
    return base, melhor, resultado


def test_escolhe_random_forest(projeto):
    *_, resultado = projeto
    assert resultado.selecao.melhor == "Random Forest"


def test_resultados_do_colab_foram_salvos(projeto):
    base, _, _ = projeto
    esperados = {
        "comparacao_modelos.csv", "dataset_utilizado.csv",
        "knn.joblib", "arvore_de_decisao.joblib", "random_forest.joblib",
        "matriz_confusao_knn.csv", "matriz_confusao_arvore_de_decisao.csv",
        "matriz_confusao_random_forest.csv",
    }
    assert esperados <= {p.name for p in (base / "resultados").iterdir()}
    assert (base / "data" / "features_passaportes.csv").exists()
    assert (base / "resultados" / "graficos" / "comparacao_modelos.png").exists()


def test_pasta_do_melhor_modelo_e_autossuficiente(projeto):
    _, melhor, _ = projeto
    for nome in ("modelo.joblib", "metadata.json", "requirements.txt",
                 "extrator_imagem.py", "predict.py", "exemplos/exemplos_teste.csv"):
        assert (melhor / nome).exists(), nome

    meta = json.loads((melhor / "metadata.json").read_text(encoding="utf-8"))
    assert meta["modelo"] == "Random Forest"
    assert len(meta["features"]) == 21
    assert meta["metricas_teste"]["n_teste"] == 75
    assert "scikit-learn==" in (melhor / "requirements.txt").read_text(encoding="utf-8")


def test_predict_roda_isolado_em_subprocesso(projeto, tmp_path):
    _, melhor, _ = projeto
    saida = subprocess.run(
        [sys.executable, str(melhor / "predict.py"), "--demo"],
        capture_output=True, text=True, cwd=tmp_path, encoding="utf-8",
    )
    assert saida.returncode == 0, saida.stderr
    assert "Acertos:" in saida.stdout


def test_predict_como_biblioteca(projeto):
    _, melhor, _ = projeto
    spec = importlib.util.spec_from_file_location("predict_isolado", melhor / "predict.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    import pandas as pd

    artefato = modulo.carregar_modelo(melhor / "modelo.joblib")
    exemplos = pd.read_csv(melhor / "exemplos" / "exemplos_teste.csv")
    previsao = modulo.classificar(artefato, exemplos)
    assert list(previsao["classe_prevista"]) == list(exemplos["label_real"])
    with pytest.raises(ValueError, match="Faltam colunas"):
        modulo.classificar(artefato, exemplos.drop(columns=["qtd_cpf"]))
