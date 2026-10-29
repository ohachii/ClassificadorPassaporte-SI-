"""Configurações centrais do projeto.

Tudo o que é "número mágico", caminho ou nome de coluna fica aqui, para que
os demais módulos não tenham valores espalhados.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos (relativos à raiz do projeto, funcionam em Windows, Linux e macOS)
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_MELHOR_MODELO = RAIZ / "melhor_modelo"

CSV_FEATURES = DIR_DADOS / "features_passaportes.csv"

# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
COLUNA_ALVO = "label"

# Colunas que NUNCA entram no treino (identificam o arquivo ou vazam a classe).
COLUNAS_NAO_TREINAVEIS = ("nome_arquivo", "caminho_arquivo", "texto_ocr")

COLUNAS_DATASET = [
    "nome_arquivo", "caminho_arquivo", "x_face", "y_face", "w_face", "h_face",
    "quantidade_palavras", "presenca_qrcode", "presenca_rostos", "qtd_cnh",
    "qtd_rg", "qtd_detran", "qtd_cpf", "qtd_documento", "qtd_identidade",
    "qtd_passaporte", "qtd_validade", "qtd_nacional", "qtd_estudante",
    "qtd_matricula", "qtd_universidade", "qtd_estadual", "qtd_une",
    "label", "texto_ocr",
]

EXEMPLOS_POR_CLASSE = 100
LINHAS_ESPERADAS = 3 * EXEMPLOS_POR_CLASSE
CLASSES_ESPERADAS = 3

# Colunas que recebem valores ausentes (NaN) aleatórios, para exercitar o Imputer.
COLUNAS_COM_NULOS = (
    "x_face", "y_face", "w_face", "h_face", "quantidade_palavras",
)
PROPORCAO_NULOS = 0.12

# --------------------------------------------------------------------------- #
# Treino / avaliação
# --------------------------------------------------------------------------- #
TAMANHO_TESTE = 0.25

# Validação cruzada usada somente para desempatar modelos tecnicamente empatados.
CV_FOLDS = 5
CV_REPETICOES = 5


# algo