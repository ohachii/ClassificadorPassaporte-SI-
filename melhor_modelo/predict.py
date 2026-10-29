"""Executa o MELHOR MODELO do classificador de documentos.

Esta pasta é autossuficiente: não depende do restante do projeto.

Uso (a partir desta pasta ou de qualquer outra):

    python predict.py --demo                    # exemplos de teste embutidos (não precisa de OCR)
    python predict.py caminho/documento.png     # classifica uma imagem (precisa de OpenCV + Tesseract)
    python predict.py --csv minhas_features.csv # classifica linhas com as 21 características
    python predict.py --csv f.csv --saida previsoes.csv

Uso como biblioteca:

    from predict import carregar_modelo, classificar
    artefato = carregar_modelo()
    classificar(artefato, dataframe_com_as_features)

AVISO: o modelo foi treinado com dados SINTÉTICOS. Não use para decisões oficiais.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

PASTA = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA / "modelo.joblib"
CAMINHO_METADATA = PASTA / "metadata.json"
CAMINHO_EXEMPLOS = PASTA / "exemplos" / "exemplos_teste.csv"

AVISO = (
    "AVISO: modelo treinado com dados SINTÉTICOS; o resultado não é confiável "
    "para documentos reais. Não use para decisões oficiais."
)


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #
def ler_metadata() -> dict:
    """Metadados do modelo (nome, métricas, versões...). Vazio se não existir."""
    if CAMINHO_METADATA.exists():
        return json.loads(CAMINHO_METADATA.read_text(encoding="utf-8"))
    return {}


def carregar_modelo(caminho: Path = CAMINHO_MODELO) -> dict:
    """Carrega ``{"modelo", "features", "classes"}`` e avisa se a versão do scikit-learn diverge."""
    if not Path(caminho).exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {caminho}")

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        artefato = joblib.load(caminho)

    if any("version" in str(a.message).lower() for a in avisos):
        import sklearn

        versao_treino = ler_metadata().get("versoes", {}).get("scikit-learn", "desconhecida")
        print(
            f"ATENÇÃO: o modelo foi treinado com scikit-learn {versao_treino}, mas você tem "
            f"{sklearn.__version__}. Instale a versão correta com "
            f"`pip install -r requirements.txt` ou gere o modelo de novo com "
            f"`python main.py treinar` (na raiz do projeto).",
            file=sys.stderr,
        )
    return artefato


# --------------------------------------------------------------------------- #
# Predição
# --------------------------------------------------------------------------- #
def classificar(artefato: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Classifica cada linha de ``features``.

    Retorna um DataFrame com a coluna ``classe_prevista`` e uma coluna
    ``prob_<classe>`` para cada classe.
    """
    esperadas = artefato["features"]
    faltando = [c for c in esperadas if c not in features.columns]
    if faltando:
        raise ValueError(f"Faltam colunas de características: {faltando}")

    entrada = features[esperadas]
    modelo = artefato["modelo"]
    resultado = pd.DataFrame({"classe_prevista": modelo.predict(entrada)}, index=entrada.index)

    probabilidades = modelo.predict_proba(entrada)
    for i, classe in enumerate(modelo.classes_):
        resultado[f"prob_{classe}"] = probabilidades[:, i].round(4)
    return resultado


def classificar_imagem(caminho_imagem, artefato: dict | None = None) -> tuple[pd.DataFrame, str]:
    """Extrai características da imagem (face + OCR) e classifica."""
    sys.path.insert(0, str(PASTA))
    from extrator_imagem import extrair_features_de_imagem

    artefato = artefato or carregar_modelo()
    features, texto = extrair_features_de_imagem(caminho_imagem)
    return classificar(artefato, features), texto


# --------------------------------------------------------------------------- #
# Linha de comando
# --------------------------------------------------------------------------- #
def _imprimir_cabecalho() -> None:
    info = ler_metadata()
    if info:
        m = info.get("metricas_teste", {})
        print(f"Modelo: {info.get('modelo')} | F1 macro no teste: {m.get('f1_macro')} "
              f"| acurácia: {m.get('acuracia')}")


def _modo_demo(artefato: dict) -> int:
    if not CAMINHO_EXEMPLOS.exists():
        print(f"Arquivo de exemplos não encontrado: {CAMINHO_EXEMPLOS}")
        return 1
    exemplos = pd.read_csv(CAMINHO_EXEMPLOS)
    previsao = classificar(artefato, exemplos)
    tabela = pd.concat([exemplos[["label_real"]], previsao], axis=1)
    tabela["acertou"] = tabela["label_real"] == tabela["classe_prevista"]
    print(tabela.to_string(index=False))
    print(f"\nAcertos: {int(tabela['acertou'].sum())}/{len(tabela)} "
          "(exemplos do conjunto de TESTE, não vistos no treino)")
    return 0


def _modo_csv(artefato: dict, caminho: str, saida: str | None) -> int:
    dados = pd.read_csv(caminho)
    previsao = classificar(artefato, dados)
    print(previsao.to_string())
    if saida:
        previsao.to_csv(saida, index_label="linha", encoding="utf-8")
        print(f"\nPrevisões salvas em: {saida}")
    return 0


def _modo_imagem(artefato: dict, caminho: str) -> int:
    previsao, texto = classificar_imagem(caminho, artefato)
    print(f"Arquivo: {caminho}")
    print("--- Texto lido pelo OCR (primeiros 300 caracteres) ---")
    print(texto[:300].strip() or "(nenhum texto reconhecido)")
    print("--- Resultado ---")
    print(f"Classe prevista: {previsao.iloc[0]['classe_prevista'].upper()}")
    for coluna in previsao.columns[1:]:
        print(f"  {coluna[5:]:<12} {previsao.iloc[0][coluna]:.2%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Executa o melhor modelo do classificador.")
    parser.add_argument("imagem", nargs="?", help="caminho de uma imagem de documento")
    parser.add_argument("--demo", action="store_true", help="roda os exemplos de teste embutidos")
    parser.add_argument("--csv", help="CSV com as colunas de características")
    parser.add_argument("--saida", help="(com --csv) arquivo CSV para salvar as previsões")
    args = parser.parse_args(argv)

    if not (args.imagem or args.demo or args.csv):
        parser.print_help()
        return 1

    artefato = carregar_modelo()
    _imprimir_cabecalho()
    try:
        if args.demo:
            codigo = _modo_demo(artefato)
        elif args.csv:
            codigo = _modo_csv(artefato, args.csv, args.saida)
        else:
            codigo = _modo_imagem(artefato, args.imagem)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1

    print(f"\n{AVISO}")
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
