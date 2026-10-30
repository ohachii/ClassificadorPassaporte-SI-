"""Orquestrador do experimento completo (Facade).

Executa, em ordem, tudo o que foi feito no notebook do Colab:

    2. gerar dataset            6. avaliar e comparar
    3. validar dados            7. gráfico comparativo
    4. separar treino/teste     8. salvar resultados e modelos
    5. treinar modelos          9. demonstração de previsão
                               10. verificar compatibilidade com o extrator

e, ao final, escolhe o melhor modelo e o exporta para ``melhor_modelo/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from classificador.avaliacao import (
    ResultadoSelecao,
    avaliar_modelos,
    escolher_melhor_modelo,
    plotar_comparacao,
    plotar_matriz_confusao,
    relatorio_classificacao,
)
from classificador.config import (
    COLUNA_ALVO,
    CSV_FEATURES,
    DIR_MELHOR_MODELO,
    DIR_RESULTADOS,
    SEED,
)
from classificador.dados import (
    contar_ausentes,
    contar_por_classe,
    dividir_treino_teste,
    gerar_dataset,
    selecionar_features,
    validar_dataset,
)
from classificador.exportacao import AVISO, exportar_melhor_modelo
from classificador.modelos import criar_modelos, treinar_modelos
from classificador.persistencia import (
    nome_arquivo_seguro,
    salvar_dataset,
    salvar_resultados,
)
from classificador.verificacao import verificar_compatibilidade


@dataclass
class ResultadoExperimento:
    """Tudo o que o experimento produziu (útil em notebooks e testes)."""

    df: pd.DataFrame
    features: list[str]
    classes: list[str]
    X_treino: pd.DataFrame
    X_teste: pd.DataFrame
    y_treino: pd.Series
    y_teste: pd.Series
    modelos: dict[str, Pipeline]
    metricas: pd.DataFrame
    predicoes: dict
    selecao: ResultadoSelecao
    arquivos_resultados: list[Path]
    arquivos_melhor_modelo: list[Path]


def _titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def demonstrar_previsao(modelo, X: pd.DataFrame, y: pd.Series, linha: int = 0) -> None:
    """Prevê uma linha do dataset. É só uma demonstração, não uma avaliação."""
    print("Classe real:", y.iloc[linha])
    print("Classe prevista:", modelo.predict(X.iloc[[linha]])[0])


def executar_experimento(
    salvar: bool = True,
    exportar: bool = True,
    mostrar_graficos: bool = False,
    pasta_resultados: Path = DIR_RESULTADOS,
    pasta_melhor_modelo: Path = DIR_MELHOR_MODELO,
    arquivo_csv: Path = CSV_FEATURES,
    pasta_dataset_imagens: Path | str | None = None,
    usar_dataset_real: bool = True,
) -> ResultadoExperimento:
    """Roda o experimento de ponta a ponta."""
    pasta_resultados = Path(pasta_resultados)
    pasta_graficos = pasta_resultados / "graficos"

    # 2. Dataset ------------------------------------------------------------ #
    _titulo("2. Dataset (Passaportes Reais / Features)")
    if pasta_dataset_imagens and Path(pasta_dataset_imagens).exists():
        from classificador.dados.extracao_dataset import extrair_dataset_de_imagens
        print(f"Extraindo características das imagens em {pasta_dataset_imagens}...")
        df = extrair_dataset_de_imagens(pasta_dataset_imagens, caminho_csv_saida=arquivo_csv if salvar else None)
    elif usar_dataset_real and Path(arquivo_csv).exists() and Path(arquivo_csv).stat().st_size > 0:
        print(f"Carregando características de {arquivo_csv}...")
        df = pd.read_csv(arquivo_csv)
    else:
        df = gerar_dataset(SEED)
        if salvar:
            salvar_dataset(df, arquivo_csv)
    print(f"Dataset em uso: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # 3. Validação ---------------------------------------------------------- #
    _titulo("3. Validação dos dados")
    features = selecionar_features(df)
    print("Quantidade por classe:")
    print(contar_por_classe(df).to_string(index=False))
    print("\nValores ausentes (resolvidos pelo Imputer):")
    print(contar_ausentes(df).to_string())
    validar_dataset(df, features, linhas_esperadas=None)
    print(f"\nCaracterísticas numéricas em uso ({len(features)}): {features}")


    # 4. Treino/teste ------------------------------------------------------- #
    _titulo("4. Separação treino/teste (estratificada)")
    X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(df, features)
    print("Treino:", X_treino.shape, "| Teste:", X_teste.shape)

    # 5. Treinamento -------------------------------------------------------- #
    _titulo("5. Treinamento")
    modelos = treinar_modelos(criar_modelos(SEED), X_treino, y_treino)

    # 6. Avaliação ---------------------------------------------------------- #
    _titulo("6. Avaliação e comparação")
    classes = sorted(df[COLUNA_ALVO].astype(str).unique())
    metricas, predicoes = avaliar_modelos(modelos, X_teste, y_teste)
    print(metricas.round(4).to_string())

    for nome, pred in predicoes.items():
        print(f"\n----- Relatório: {nome} -----")
        print(relatorio_classificacao(y_teste, pred))
        plotar_matriz_confusao(
            y_teste, pred, classes, nome,
            salvar_em=pasta_graficos / f"matriz_confusao_{nome_arquivo_seguro(nome)}.png" if salvar else None,
            mostrar=mostrar_graficos,
        )

    # 7. Gráfico comparativo ------------------------------------------------ #
    plotar_comparacao(
        metricas,
        salvar_em=pasta_graficos / "comparacao_modelos.png" if salvar else None,
        mostrar=mostrar_graficos,
    )

    # 8. Salvar resultados -------------------------------------------------- #
    arquivos_resultados: list[Path] = []
    if salvar:
        _titulo("8. Resultados salvos")
        arquivos_resultados = salvar_resultados(
            pasta_resultados, df, metricas, modelos, predicoes, y_teste, features, classes
        )
        print("Pasta:", pasta_resultados)
        for arquivo in arquivos_resultados:
            print("-", arquivo.name)

    # Escolha do melhor modelo ---------------------------------------------- #
    _titulo("Escolha do melhor modelo")
    selecao = escolher_melhor_modelo(
        metricas, modelos, X_treino, y_treino, n_teste=len(X_teste), seed=SEED
    )
    print(selecao.finalistas.round(4).to_string(index=False))
    print("\n" + selecao.justificativa)
    print(f"\nMELHOR MODELO: {selecao.melhor}")

    # 9. Demonstração ------------------------------------------------------- #
    _titulo("9. Demonstração de previsão (uma linha do dataset)")
    demonstrar_previsao(modelos[selecao.melhor], df[features], df[COLUNA_ALVO].astype(str))

    # Exportação ------------------------------------------------------------ #
    arquivos_melhor: list[Path] = []
    if exportar:
        _titulo(f"Exportação do melhor modelo -> {pasta_melhor_modelo}")
        linha = metricas[metricas["modelo"] == selecao.melhor].iloc[0].to_dict()
        arquivos_melhor = exportar_melhor_modelo(
            selecao.melhor, modelos[selecao.melhor], features, classes, linha,
            selecao, X_treino, X_teste, y_teste, destino=pasta_melhor_modelo,
        )
        for arquivo in arquivos_melhor:
            print("-", arquivo.relative_to(pasta_melhor_modelo))

        # 10. Compatibilidade com o extrator de imagens ---------------------- #
        _titulo("10. Compatibilidade modelo <-> extrator de imagens")
        verificar_compatibilidade(pasta_melhor_modelo / "modelo.joblib")

    print(f"\n{AVISO}")
    return ResultadoExperimento(
        df, features, classes, X_treino, X_teste, y_treino, y_teste,
        modelos, metricas, predicoes, selecao, arquivos_resultados, arquivos_melhor,
    )
