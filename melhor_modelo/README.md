# Melhor modelo — Random Forest

Esta pasta é **autossuficiente**: dá para copiá-la para qualquer lugar e executá-la sem o restante do projeto.
Ela contém o melhor classificador encontrado no experimento (veja o `README.md` da raiz) e tudo o que é
preciso para usá-lo.

| Arquivo | Para que serve |
|---|---|
| `predict.py` | **Programa para executar o modelo** (linha de comando e biblioteca) |
| `modelo.joblib` | Random Forest treinado (`{"modelo", "features", "classes"}`) |
| `metadata.json` | Métricas, critério de escolha, características esperadas e versões das bibliotecas |
| `extrator_imagem.py` | Transforma a imagem de um documento nas 21 características (rosto + OCR) |
| `requirements.txt` | Dependências, com a versão exata do scikit-learn usada no treino |
| `exemplos/exemplos_teste.csv` | 9 exemplos do conjunto de **teste** (3 por classe), com a classe real |
| `exemplos/documento_exemplo_estudante.png` | Imagem **fictícia** para demonstrar a leitura por OCR |

## Desempenho (conjunto de teste, 75 exemplos)

Acurácia **0,96** · Precisão macro **0,9643** · Recall macro **0,96** · F1 macro **0,9599**

O modelo foi escolhido pelo desempate por validação cruzada com a Árvore de Decisão (empate técnico no teste).
O detalhe está em `metadata.json` → `selecao`.

## Como executar

```bash
pip install -r requirements.txt
```

**Demonstração (não precisa de OCR nem de imagem):**

```bash
python predict.py --demo
```

Classifica os 9 exemplos de teste e mostra classe real × prevista e a probabilidade de cada classe.

**Classificar a imagem de um documento** (precisa do programa Tesseract; veja abaixo):

```bash
python predict.py exemplos/documento_exemplo_estudante.png
python predict.py "C:\Users\Maria\Área de Trabalho\documento.png"
```

**Classificar linhas já extraídas** (CSV com as 21 colunas listadas em `metadata.json`):

```bash
python predict.py --csv minhas_features.csv --saida previsoes.csv
```

Valores ausentes (vazios) no CSV são aceitos: o próprio modelo os preenche com a mediana do treino.

### Usar em Python

```python
from predict import carregar_modelo, classificar, classificar_imagem

artefato = carregar_modelo()
previsao = classificar(artefato, dataframe_com_as_21_colunas)   # classe_prevista + prob_<classe>
previsao, texto_ocr = classificar_imagem("documento.png", artefato)
```

### Tesseract (só para imagens)

| Sistema | Instalação |
|---|---|
| Windows | instalador do [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) com o idioma *Portuguese*; depois `$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"` |
| Ubuntu/Debian | `sudo apt install tesseract-ocr tesseract-ocr-por` |
| macOS | `brew install tesseract tesseract-lang` |

`TESSERACT_LANG` troca o idioma do OCR (padrão `por`).

## Classes de saída

`brasileiro` · `estrangeiro` · `invalido`

## Problemas comuns

- **"ATENÇÃO: o modelo foi treinado com scikit-learn X, mas você tem Y"** — o `.joblib` depende da versão
  do scikit-learn. Instale a versão do `requirements.txt` ou gere o modelo de novo, na raiz do projeto:
  `python main.py treinar` (isso atualiza esta pasta).
- **"Tesseract não encontrado"** — instale o programa e/ou defina `TESSERACT_CMD`.
- **"O idioma 'por' do Tesseract não está instalado"** — instale o pacote de idioma português.

## Atenção

O modelo foi treinado com **dados sintéticos**. O desempenho acima não vale para documentos reais, e o
modelo pode errar com alta confiança em entradas fora do padrão (por exemplo, imagens em que nenhum rosto
ou termo é reconhecido tendem a ser classificadas como `estrangeiro`).
**Não use para decisões oficiais.**

> `modelo.joblib`, `metadata.json`, `requirements.txt`, `extrator_imagem.py` e `exemplos/exemplos_teste.csv`
> são gerados por `python main.py treinar` na raiz do projeto. `predict.py` e este README são mantidos à mão.
