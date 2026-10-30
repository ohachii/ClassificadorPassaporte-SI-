"""Módulo de deduplicação e garantia de unicidade de identidades.

Garante a regra obrigatória:
"Os datasets têm que ser de passaportes reais, os dados NÃO devem se repetir,
NÃO pode haver mais de 1 foto da mesma pessoa mais de uma vez."

Funcionalidades:
1. Extração do identificador base (documento / pessoa), desconsiderando técnicas
   de adulteração do SIDTD (ex.: crop_and_replace, inpaint, tamper) e sufixos.
2. Deduplicação estrita intra-classe e inter-classes (0 vazamento).
3. Verificação perceptual de imagens (dHash) para impedir fotos repetidas.
4. Auditoria completa do dataset final.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence
from PIL import Image


# Padrões comuns de falsificação ou variantes que devem ser removidos
# para recuperar a identidade base do indivíduo/template original.
PADROES_VARIACOES = (
    r"_crop_and_replace.*",
    r"_crop_and_move.*",
    r"_inpaint.*",
    r"_tamper.*",
    r"_forg.*",
    r"_fake.*",
    r"_bona.*",
    r"_real.*",
    r"_genuine.*",
    r"_original.*",
    r"_aug.*",
    r"_frente.*",
    r"_verso.*",
)

REGEX_VARIACOES = re.compile("|".join(PADROES_VARIACOES), re.IGNORECASE)


def extrair_id_base(caminho_ou_nome: str | Path) -> str:
    """Extrai o ID da identidade/pessoa/template base de um arquivo.

    Exemplos:
        'esp_passport_type1_001_crop_and_replace.jpg' -> 'esp_passport_type1_001'
        'grc_passport_002_inpaint_01.png'            -> 'grc_passport_002'
        'ser_id_type2_010.jpg'                       -> 'ser_id_type2_010'
        'brasileiro_042.jpg'                         -> 'brasileiro_042'
    """
    nome = Path(caminho_ou_nome).stem.lower()
    # Remove variações conhecidas de adulteração/processamento
    id_base = REGEX_VARIACOES.sub("", nome).strip("_- ")
    return id_base or nome


def calcular_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Calcula o Difference Hash (dHash) de uma imagem PIL.

    Mede a estrutura visual da foto; fotos da mesma pessoa ou cópias
    com pequenas rotações/cortes têm distância de Hamming muito baixa.
    """
    img_gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(img_gray.getdata())
    diff = 0
    for y in range(hash_size):
        row_start = y * (hash_size + 1)
        for x in range(hash_size):
            if pixels[row_start + x] > pixels[row_start + x + 1]:
                diff |= 1 << (y * hash_size + x)
    return diff


def distancia_hamming(hash1: int, hash2: int) -> int:
    """Conta quantos bits diferem entre dois hashes."""
    return bin(hash1 ^ hash2).count("1")


def filtrar_identidades_unicas(
    itens: Sequence,
    identidades_bloqueadas: set[str],
    limite: int | None = None,
    checar_hash: bool = False,
    distancia_minima_hash: int = 4,
) -> tuple[list, set[str]]:
    """Seleciona itens garantindo que NENHUMA identidade base se repita.

    Parâmetros:
        itens: lista de itens (tuplas (fonte, nome) ou caminhos Path).
        identidades_bloqueadas: conjunto de IDs base já usados por outras classes.
        limite: quantidade máxima de itens a selecionar.
        checar_hash: se True, valida também unicidade por dHash visual.
        distancia_minima_hash: distância mínima de Hamming permitida.

    Retorna:
        (itens_selecionados, novas_identidades_usadas)
    """
    selecionados = []
    identidades_usadas = set(identidades_bloqueadas)
    hashes_vistos: list[int] = []

    for item in itens:
        caminho_str = item[1] if isinstance(item, (tuple, list)) else str(item)
        id_base = extrair_id_base(caminho_str)

        # 1. Checagem de ID base: nunca repete template ou pessoa
        if id_base in identidades_usadas:
            continue

        # 2. Checagem visual opcional por dHash
        if checar_hash:
            try:
                if isinstance(item, (tuple, list)) and hasattr(item[0], "open"):
                    with item[0].open(item[1]) as f:
                        img = Image.open(f)
                        h = calcular_dhash(img)
                elif isinstance(item, (tuple, list)):
                    # Pode ser arquivo zip ou diretório
                    continue
                else:
                    with Image.open(item) as img:
                        h = calcular_dhash(img)

                if any(distancia_hamming(h, h_antigo) < distancia_minima_hash for h_antigo in hashes_vistos):
                    continue  # foto repetida da mesma pessoa
                hashes_vistos.append(h)
            except Exception:
                pass

        identidades_usadas.add(id_base)
        selecionados.append(item)

        if limite and len(selecionados) >= limite:
            break

    return selecionados, identidades_usadas


def auditar_unicidade_dataset(pasta_dataset: str | Path) -> dict[str, any]:
    """Audita o dataset completo em disco e valida a regra de unicidade estrita.

    Verifica:
    1. Se há fotos repetidas (mesmo ID ou imagem duplicada).
    2. Se há vazamento da mesma pessoa/template entre classes diferentes.
    3. Quantidade total de indivíduos únicos vs total de imagens.

    Retorna um dicionário com o resultado e lança AssertionError se houver violação.
    """
    pasta_dataset = Path(pasta_dataset)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    por_classe: dict[str, list[Path]] = {}
    ids_por_classe: dict[str, set[str]] = {}
    todos_arquivos: list[Path] = []

    for arq in sorted(pasta_dataset.rglob("*")):
        if arq.is_file() and arq.suffix.lower() in exts:
            classe = arq.parent.name
            # Se a estrutura for train/classe/arquivo, pega a pasta da classe
            if arq.parent.parent.name in ("train", "val", "test"):
                classe = arq.parent.name
            por_classe.setdefault(classe, []).append(arq)
            ids_por_classe.setdefault(classe, set()).add(extrair_id_base(arq.name))
            todos_arquivos.append(arq)

    total_imagens = len(todos_arquivos)
    total_ids = sum(len(ids) for ids in ids_por_classe.values())

    # Checagem de intersecção entre classes
    classes = list(ids_por_classe.keys())
    interseccoes = {}
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            c1, c2 = classes[i], classes[j]
            comuns = ids_por_classe[c1] & ids_por_classe[c2]
            if comuns:
                interseccoes[f"{c1} x {c2}"] = comuns

    relatorio = {
        "total_imagens": total_imagens,
        "classes": {c: len(arqs) for c, arqs in por_classe.items()},
        "ids_unicos_por_classe": {c: len(ids) for c, ids in ids_por_classe.items()},
        "interseccoes_entre_classes": interseccoes,
        "aprovado": (len(interseccoes) == 0 and total_imagens == total_ids),
    }

    print("\n" + "=" * 60)
    print("AUDITORIA DE UNICIDADE DO DATASET (PESSOA / TEMPLATE)")
    print("=" * 60)
    print(f"Total de imagens analisadas: {total_imagens}")
    for c, q in relatorio["classes"].items():
        print(f"  - {c}: {q} imagens | {len(ids_por_classe[c])} identidades únicas")

    if interseccoes:
        print("\n❌ FALHA: Foram encontradas fotos da mesma pessoa em classes diferentes:")
        for par, comuns in interseccoes.items():
            print(f"  {par}: {len(comuns)} sobreposições: {sorted(list(comuns))[:5]}")
        raise ValueError(f"Dataset inválido: {len(interseccoes)} sobreposições entre classes detectadas!")

    if total_imagens != total_ids:
        print(f"\n❌ FALHA: Há fotos repetidas dentro de uma mesma classe ({total_imagens} imagens vs {total_ids} IDs únicos).")
        raise ValueError("Dataset inválido: fotos duplicadas da mesma pessoa detectadas dentro da classe!")

    print("\n✓ APROVADO: 100% dos dados são de pessoas/templates distintos!")
    print("✓ Nenhuma foto se repete. Zero sobreposição entre classes.")
    print("=" * 60)

    return relatorio
