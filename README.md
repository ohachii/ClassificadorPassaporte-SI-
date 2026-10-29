# Classificador de Documentos

**Disciplina:** Sistemas Inteligentes

Classifica documentos em três classes — `brasileiro`, `estrangeiro` e `invalido` — comparando
**KNN**, **Árvore de Decisão** e **Random Forest**. O projeto é a versão organizada para VS Code do
notebook do Google Colab, e entrega duas coisas:

1. **Tudo o que foi feito no Colab**, separado em módulos (pasta `classificador/`) e reproduzível com um comando.
2. **O melhor modelo, em pasta separada** (`melhor_modelo/`), pronto para executar sozinho.

> ⚠️ **Limitação importante:** o dataset é **sintético/fictício**. As métricas descrevem somente esse
> conjunto e **não** comprovam desempenho em documentos reais. Não use para decisões oficiais.

---

## Resultado

Avaliação no conjunto de teste (75 exemplos, 25 por classe), mesma divisão do Colab:

| Modelo | Acurácia | Precisão macro | Recall macro | F1 macro |
|---|---|---|---|---|
| Árvore de Decisão | 0,96 | 0,9605 | 0,96 | 0,9600 |
| **Random Forest** ⭐ | 0,96 | 0,9643 | 0,96 | 0,9599 |
| KNN | 0,92 | 0,9274 | 0,92 | 0,9207 |

**Melhor modelo: Random Forest.** A Árvore e a Random Forest empatam tecnicamente no teste (a diferença
de F1 é de 0,0001, e um único exemplo de teste já vale ~1,3 ponto percentual). O desempate usa validação
cruzada **somente nos dados de treino**, onde a Random Forest vai melhor (F1 médio 0,979 contra 0,949).
O critério completo está em [`classificador/avaliacao/selecao.py`](classificador/avaliacao/selecao.py).

---

## Estrutura do projeto

```
classificador-documentos/
├── main.py                      # ponto de entrada: python main.py treinar | verificar
├── requirements.txt             # dependências do projeto
├── requirements-dev.txt         # + pytest e ipykernel (testes e notebooks)
│
├── classificador/               # pacote com todo o código (organizado por responsabilidade)
│   ├── config.py                # caminhos, SEED, nomes de colunas, constantes
│   ├── dados/                   # gerador.py · validacao.py · divisao.py
│   ├── modelos/                 # fabrica.py (Factory dos 3 modelos) · treinamento.py
│   ├── avaliacao/               # metricas.py · graficos.py · selecao.py (melhor modelo)
│   ├── extracao/imagem.py       # visão computacional + OCR (imagem -> 21 características)
│   ├── persistencia.py          # salvar/carregar dataset, modelos e resultados
│   ├── verificacao.py           # compatibilidade modelo <-> extrator
│   ├── exportacao.py            # gera a pasta melhor_modelo/
│   └── experimento.py           # orquestrador (Facade): roda tudo em ordem
│
├── melhor_modelo/               # ⭐ MELHOR MODELO, autossuficiente (veja o README dela)
│   ├── predict.py               #    python predict.py --demo | imagem.png | --csv arquivo.csv
│   ├── modelo.joblib            #    Random Forest treinado
│   ├── metadata.json            #    métricas, critério de escolha, versões das bibliotecas
│   ├── extrator_imagem.py       #    extrator de características de imagens
│   ├── requirements.txt         #    dependências mínimas para executar
│   └── exemplos/                #    exemplos de teste e uma imagem de demonstração
│
├── notebooks/
│   ├── Classificador_Documentos_Integrado_Colab.ipynb   # ORIGINAL do Colab, sem alterações
│   └── 01_experimento_vscode.ipynb                      # mesmo experimento, chamando os módulos
│
├── data/features_passaportes.csv    # dataset sintético gerado (seção 2 do Colab)
├── resultados/                      # saídas do experimento (seção 8 do Colab)
│   ├── comparacao_modelos.csv · dataset_utilizado.csv
│   ├── knn.joblib · arvore_de_decisao.joblib · random_forest.joblib
│   ├── matriz_confusao_*.csv
│   └── graficos/                    # matrizes de confusão e comparação (PNG)
├── tests/                           # testes automatizados (pytest)
└── .vscode/                         # configurações e atalhos F5 para o VS Code
```

### Do notebook do Colab para os módulos

| Seção do Colab | Onde está agora |
|---|---|
| 1. Preparar o ambiente | `requirements.txt` |
| 2. Dataset sintético | `classificador/dados/gerador.py` |
| 3. Validar os dados | `classificador/dados/validacao.py` |
| 4. Treino e teste | `classificador/dados/divisao.py` |
| 5. Treinar os classificadores | `classificador/modelos/` |
| 6. Avaliar e comparar | `classificador/avaliacao/metricas.py` e `graficos.py` |
| 7. Gráfico comparativo | `classificador/avaliacao/graficos.py` |
| 8. Salvar resultados e modelos | `classificador/persistencia.py` |
| 9. Demonstração de previsão | `classificador/experimento.py` |
| 9.1. Imagens reais (OpenCV + OCR) | `classificador/extracao/imagem.py` |
| 10. Verificação de colunas | `classificador/verificacao.py` |
| 11. Conclusão e limitações | este README e o notebook `01_experimento_vscode.ipynb` |
| *(novo)* Escolha do melhor modelo | `classificador/avaliacao/selecao.py` + `classificador/exportacao.py` |

O dataset e as métricas são **idênticos** aos do Colab (mesma `SEED = 42`, mesma ordem de geração).
Mudanças em relação ao Colab, todas pequenas: caminhos `/content/...` viraram caminhos relativos ao
projeto; os nomes dos arquivos `.joblib` ficam sem acento (`arvore_de_decisao.joblib`); `assert`
virou `ValueError` com mensagem; o extrator de imagens lê arquivos em pastas com acento (comum no Windows).

---

## Como rodar no VS Code

**1. Abra a pasta** `classificador-documentos` no VS Code (*File → Open Folder*) e instale as extensões
recomendadas (o VS Code sugere sozinho: *Python* e *Jupyter*).

**2. Crie o ambiente virtual e instale as dependências** (terminal do VS Code, `` Ctrl+` ``):

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements-dev.txt
```

Se o VS Code perguntar, selecione o interpretador `.venv` (*Ctrl+Shift+P → Python: Select Interpreter*).

**3. Reproduza o experimento do Colab:**

```bash
python main.py treinar
```

Isso executa as seções 2 a 10, grava tudo em `resultados/` e `data/`, escolhe o melhor modelo e atualiza
a pasta `melhor_modelo/`. Gráficos são salvos em `resultados/graficos/` (use `--mostrar-graficos` para
abri-los também em janelas).

**4. Ou abra o notebook** `notebooks/01_experimento_vscode.ipynb`, escolha o kernel `.venv` e use *Run All*.

**5. Execute o melhor modelo:**

```bash
python melhor_modelo/predict.py --demo
```

No VS Code, também dá para usar **F5** e escolher uma das 3 configurações prontas.

**6. Testes automatizados:**

```bash
python -m pytest
```

### Tesseract (somente para classificar imagens)

O treinamento e o `--demo` **não precisam** do Tesseract. Para ler texto de imagens é preciso instalar o
programa (o pacote Python `pytesseract` é só a ponte) **com o idioma português**:

| Sistema | Instalação |
|---|---|
| Windows | instalador do [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki); marque *Portuguese* nos idiomas. Depois defina o caminho: `$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"` |
| Ubuntu/Debian | `sudo apt install tesseract-ocr tesseract-ocr-por` |
| macOS | `brew install tesseract tesseract-lang` |

Variáveis de ambiente opcionais: `TESSERACT_CMD` (caminho do executável, se não estiver no PATH) e
`TESSERACT_LANG` (idioma do OCR; padrão `por`).

---

## Organização e padrões de projeto

- **Separação por responsabilidade:** dados, modelos, avaliação, extração e persistência são pacotes
  independentes; nenhum arquivo mistura tarefas.
- **Factory** (`modelos/fabrica.py`): cada algoritmo tem uma função de criação registrada em um dicionário.
  Para adicionar um modelo, escreva a função e registre — o restante do código não muda.
- **Pipeline** (scikit-learn): imputação e padronização ficam *dentro* do modelo, ajustadas só com dados
  de treino (sem vazamento do teste).
- **Facade** (`experimento.py`): uma única função (`executar_experimento`) orquestra todas as etapas.
- **Configuração centralizada** (`config.py`): sem números mágicos nem caminhos espalhados.
- **Artefato autossuficiente** (`melhor_modelo/`): gerado por `exportacao.py`, roda sem o resto do projeto.

---

## Limitações conhecidas

- **Dados sintéticos.** O modelo nunca viu um documento real. Várias características refletem os padrões
  usados na geração dos exemplos (por exemplo, todo "inválido" tem `qtd_documento = 1`), o que facilita
  artificialmente a classificação.
- **Pipeline de imagem não validado.** O extrator (OpenCV + OCR) existe e funciona, mas o modelo foi
  treinado com dados sintéticos, não com saídas do extrator. Em testes informais, uma imagem fictícia de
  carteira de estudante foi classificada como `invalido`, porém um passaporte brasileiro fictício **sem
  rosto detectável** foi classificado como `estrangeiro`. Isso mostra a dependência do modelo da detecção
  de rosto.
- **Probabilidades pouco confiáveis fora do padrão.** Entradas com quase nenhuma característica
  reconhecida (por exemplo, todas zeradas) tendem a ser classificadas como `estrangeiro` com alta
  probabilidade.
- **Contagem de termos por trecho.** O extrator conta ocorrências por substring (como no Colab); termos
  curtos como `rg` e `une` também casam dentro de outras palavras.
- **QR Code** não é detectado ainda (`presenca_qrcode` é sempre 0 para imagens reais).
- **Conjunto de teste pequeno** (75 exemplos): diferenças de 1 a 2 pontos percentuais não são significativas.
