"""Pipeline de extração de características de imagens reais (seção 9.1 do notebook).

Recebe a imagem de um documento e gera a linha de atributos que o modelo espera:

* detecção de rosto com OpenCV (Haar Cascade);
* texto via OCR (Tesseract, idioma português);
* contagem de termos de interesse no texto extraído.

ATENÇÃO: este módulo é AUTOCONTIDO de propósito (usa só bibliotecas externas,
nunca outros módulos do projeto). Ele é copiado para ``melhor_modelo/`` quando o
melhor modelo é exportado, para que essa pasta rode sozinha.

Requisitos (apenas para processar imagens):
    pip install opencv-python-headless pytesseract
    + o programa Tesseract instalado no sistema, com o idioma "por".

Variáveis de ambiente opcionais:
    TESSERACT_CMD   caminho do executável do Tesseract, se ele não estiver no PATH
                    (Windows: C:\\Program Files\\Tesseract-OCR\\tesseract.exe)
    TESSERACT_LANG  idioma do OCR (padrão: "por")
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

IDIOMA_OCR = os.environ.get("TESSERACT_LANG", "por")

# Termos buscados no texto do OCR. A característica vale a SOMA das ocorrências
# de todos os termos da tupla (mesma regra do notebook).
TERMOS_POR_FEATURE: dict[str, tuple[str, ...]] = {
    "qtd_cnh": ("cnh", "habilitação", "habilitacao"),
    "qtd_rg": ("rg", "registro geral"),
    "qtd_detran": ("detran",),
    "qtd_cpf": ("cpf",),
    "qtd_documento": ("documento", "document"),
    "qtd_identidade": ("identidade", "identity"),
    "qtd_passaporte": ("passaporte", "passport", "pasaporte", "passeport"),
    "qtd_validade": ("validade", "valido", "expiry", "expiration"),
    "qtd_nacional": (
        "nacional", "nacionalidade", "nationality", "brasil", "brazil",
        "federativa", "republica", "república"
    ),
    "qtd_estudante": ("estudante", "carteira", "estudantil", "student"),
    "qtd_matricula": ("matricula", "matrícula"),
    "qtd_universidade": ("universidade", "university"),
    "qtd_estadual": ("estadual",),
    "qtd_une": ("une",),
}

FEATURES_BASE = [
    "x_face", "y_face", "w_face", "h_face", "quantidade_palavras",
    "presenca_qrcode", "presenca_rostos",
]

# Ordem exata das colunas geradas (deve ser igual à ordem usada no treino).
FEATURES_EXTRATOR = FEATURES_BASE + list(TERMOS_POR_FEATURE)


def _importar_dependencias():
    """Importa OpenCV e pytesseract só quando necessário, com erro amigável."""
    try:
        import cv2
        import pytesseract
    except ImportError as erro:
        raise ImportError(
            "Para processar imagens instale: pip install opencv-python-headless pytesseract"
        ) from erro

    comando = os.environ.get("TESSERACT_CMD")
    if comando:
        pytesseract.pytesseract.tesseract_cmd = comando
    return cv2, pytesseract


def _carregar_imagem(cv2, caminho: Path):
    """Lê a imagem; funciona também com acentos no caminho (Windows)."""
    if not caminho.exists():
        raise FileNotFoundError(f"Imagem não encontrada no caminho: {caminho}")
    # np.fromfile + imdecode evita o problema do cv2.imread com caminhos não-ASCII.
    img = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível ler o arquivo como imagem: {caminho}")
    return img


def _detectar_face(cv2, img) -> tuple[float, float, float, float, float]:
    """Retorna ``(presenca, x, y, w, h)`` da primeira face detectada."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascata = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascata.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    x, y, w, h = map(float, faces[0])
    return 1.0, x, y, w, h


def _detectar_qrcode(cv2, img) -> float:
    """Verifica se há QR Code detectável na imagem."""
    try:
        detector = cv2.QRCodeDetector()
        val, _, _ = detector.detectAndDecode(img)
        return 1.0 if val else 0.0
    except Exception:
        return 0.0


def _ler_texto(pytesseract, img) -> str:
    """Extrai texto com OCR, tentando idioma configurado e fallbacks."""
    for lang in (IDIOMA_OCR, "por+eng", "por", "eng"):
        try:
            return pytesseract.image_to_string(img, lang=lang)
        except Exception:
            continue
    return ""


def contar_termos(texto: str) -> dict[str, float]:
    """Conta os termos de interesse (busca por trecho, em minúsculas)."""
    texto = texto.lower()
    return {
        feature: float(sum(texto.count(termo) for termo in termos))
        for feature, termos in TERMOS_POR_FEATURE.items()
    }


def extrair_features_de_imagem(caminho_imagem) -> tuple[pd.DataFrame, str]:
    """Processa a imagem de um documento.

    Retorna ``(features, texto_ocr)``: um DataFrame de 1 linha com as colunas
    de :data:`FEATURES_EXTRATOR` e o texto bruto lido pelo OCR.
    """
    cv2, pytesseract = _importar_dependencias()
    img = _carregar_imagem(cv2, Path(caminho_imagem))

    presenca, x, y, w, h = _detectar_face(cv2, img)
    qrcode = _detectar_qrcode(cv2, img)
    texto = _ler_texto(pytesseract, img)

    features = {
        "x_face": x,
        "y_face": y,
        "w_face": w,
        "h_face": h,
        "quantidade_palavras": float(len(texto.lower().split())),
        "presenca_qrcode": qrcode,
        "presenca_rostos": presenca,
        **contar_termos(texto),
    }
    return pd.DataFrame([features], columns=FEATURES_EXTRATOR), texto

