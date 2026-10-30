"""Testes para o motor de deduplicação e garantia de fotos/pessoas únicas."""

import pytest
from pathlib import Path

from classificador.dados.deduplicacao import (
    extrair_id_base,
    filtrar_identidades_unicas,
    auditar_unicidade_dataset,
)


def test_extrair_id_base_remove_variacoes_e_adulteracoes():
    exemplos = {
        "esp_passport_type1_001_crop_and_replace.jpg": "esp_passport_type1_001",
        "grc_passport_002_inpaint.png": "grc_passport_002",
        "ser_passport_003_tamper_01.jpg": "ser_passport_003",
        "alb_id_type1_005_fake.jpg": "alb_id_type1_005",
        "fin_passport_006_bona_fide.jpg": "fin_passport_006",
        "br_aug_007.jpg": "br",
        "brasileiro_001.jpg": "brasileiro_001",
    }
    for arquivo, esperado in exemplos.items():
        assert extrair_id_base(arquivo) == esperado


def test_filtrar_identidades_unicas_impede_duplicatas_e_vazamento():
    candidatos_estrangeiro = [
        ("zip", "esp_passport_001.jpg"),
        ("zip", "esp_passport_001_bona.jpg"),  # mesma pessoa/template
        ("zip", "grc_passport_002.jpg"),
        ("zip", "ser_passport_003.jpg"),
    ]
    selecionados_est, ids_usados = filtrar_identidades_unicas(
        candidatos_estrangeiro,
        identidades_bloqueadas=set(),
        limite=10,
    )
    # Apenas 3 pessoas diferentes
    assert len(selecionados_est) == 3
    assert ids_usados == {"esp_passport_001", "grc_passport_002", "ser_passport_003"}

    # Tentativa de colocar fakes das mesmas pessoas em inválido:
    candidatos_invalido = [
        ("zip", "esp_passport_001_crop_and_replace.jpg"),  # repetido de estrangeiro!
        ("zip", "grc_passport_002_inpaint.jpg"),           # repetido de estrangeiro!
        ("zip", "est_id_004_crop_and_replace.jpg"),        # nova pessoa
    ]
    selecionados_inv, todos_ids = filtrar_identidades_unicas(
        candidatos_invalido,
        identidades_bloqueadas=ids_usados,
        limite=10,
    )
    # Apenas a pessoa nova pode entrar! As outras foram bloqueadas para evitar vazamento
    assert len(selecionados_inv) == 1
    assert "est_id_004" in todos_ids
    assert "esp_passport_001" not in [extrair_id_base(item[1]) for item in selecionados_inv]


def test_auditar_unicidade_dataset(tmp_path):
    pasta_ds = tmp_path / "dataset"
    (pasta_ds / "train" / "brasileiro").mkdir(parents=True)
    (pasta_ds / "train" / "estrangeiro").mkdir(parents=True)
    (pasta_ds / "train" / "invalido").mkdir(parents=True)

    (pasta_ds / "train" / "brasileiro" / "br_001.jpg").write_text("a")
    (pasta_ds / "train" / "estrangeiro" / "esp_002.jpg").write_text("b")
    (pasta_ds / "train" / "invalido" / "doc_003.jpg").write_text("c")

    relatorio = auditar_unicidade_dataset(pasta_ds)
    assert relatorio["aprovado"] is True
    assert relatorio["total_imagens"] == 3

    # Simulando vazamento entre classes (mesma pessoa em estrangeiro e invalido)
    (pasta_ds / "train" / "invalido" / "esp_002_crop_and_replace.jpg").write_text("d")
    with pytest.raises(ValueError, match="sobreposições entre classes"):
        auditar_unicidade_dataset(pasta_ds)
