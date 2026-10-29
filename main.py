"""Ponto de entrada do projeto.

Uso:
    python main.py treinar                  # reproduz tudo o que foi feito no Colab
    python main.py treinar --mostrar-graficos
    python main.py verificar                # confere modelo x extrator de imagens

Para EXECUTAR o melhor modelo (classificar imagens), use a pasta separada:
    python melhor_modelo/predict.py --demo
"""

from __future__ import annotations

import argparse
import sys


def _configurar_saida_utf8() -> None:
    """Evita erro de acentuação quando a saída é redirecionada no Windows."""
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")


def _treinar(args: argparse.Namespace) -> int:
    if not args.mostrar_graficos:
        import matplotlib

        matplotlib.use("Agg")  # sem janelas: os gráficos são só salvos em resultados/graficos

    from classificador.experimento import executar_experimento

    executar_experimento(
        salvar=True,
        exportar=not args.sem_exportar,
        mostrar_graficos=args.mostrar_graficos,
    )
    return 0


def _verificar(_: argparse.Namespace) -> int:
    from classificador.config import DIR_MELHOR_MODELO
    from classificador.verificacao import verificar_compatibilidade

    caminho = DIR_MELHOR_MODELO / "modelo.joblib"
    if not caminho.exists():
        print(f"Modelo não encontrado em {caminho}. Rode antes: python main.py treinar")
        return 1
    return 1 if verificar_compatibilidade(caminho) else 0


def main(argv: list[str] | None = None) -> int:
    _configurar_saida_utf8()
    parser = argparse.ArgumentParser(description="Classificador de documentos")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_treinar = sub.add_parser("treinar", help="roda o experimento completo e exporta o melhor modelo")
    p_treinar.add_argument("--mostrar-graficos", action="store_true",
                           help="abre os gráficos em janelas (além de salvá-los)")
    p_treinar.add_argument("--sem-exportar", action="store_true",
                           help="não atualiza a pasta melhor_modelo/")
    p_treinar.set_defaults(funcao=_treinar)

    p_verificar = sub.add_parser("verificar", help="confere a compatibilidade modelo x extrator")
    p_verificar.set_defaults(funcao=_verificar)

    args = parser.parse_args(argv)
    return args.funcao(args)


if __name__ == "__main__":
    raise SystemExit(main())
