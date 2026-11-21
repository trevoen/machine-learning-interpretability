# Detecção de toxicidade no chat de League of Legends

Trabalho final de PLN — detecção de discurso de ódio/toxicidade em mensagens de
chat de partidas de League of Legends, com fine-tuning de Transformers e uma
análise original cruzando toxicidade com o contexto do autor da mensagem: o
arquétipo do campeão que ele joga (via Data Dragon, API pública da Riot Games)
e sua relação com o jogador denunciado (ofensor, aliado ou inimigo).

## Estrutura

```
src/
  config.py     caminhos e hiperparâmetros
  data.py       carregamento/limpeza e folds agrupados por partida
  riot_api.py   integração com a Data Dragon (dados estáticos de campeões)
  train.py      fine-tuning comparativo (DistilBERT / BERT / RoBERTa) com CV
  baseline.py   baseline zero-shot (unitary/toxic-bert) nos mesmos folds
  compare.py    bootstrap pareado por partida, subgrupos e análise de erros
  evaluate.py   avaliação no teste, tabela comparativa e métricas por subgrupo
  analysis.py   contribuição própria: toxicidade x papel do campeão e x relação
                com o ofensor (qui-quadrado, V de Cramér, resíduos)
  demo.py       demo interativa de linha de comando
  inference.py  carregamento dos modelos treinados e predição (playground/Colab)
app/
  playground.py playground Streamlit comparando os três modelos
colab/
  train_colab.ipynb  treino no Google Colab + playground Gradio
outputs/        modelos treinados, métricas e gráficos (gerado, não versionado)
```

## Pré-requisitos

Este projeto espera as seguintes pastas ao lado dele (`D:\Downloads\PLN\`):
- `esorics26_toxicity/` — repositório oficial do paper que produziu o dataset
  L2DTnH (clonado de https://github.com/irdin-pekaric/esorics26_toxicity).
- `datasetlol/` — dados brutos do Kaggle "League of Legends Tribunal Chatlogs"
  (https://www.kaggle.com/datasets/simshengxue/league-of-legends-tribunal-chatlogs),
  usados como contexto adicional (não como fonte do rótulo de toxicidade).

```bash
python -m venv .venv
# PyTorch com CUDA (GPU NVIDIA); sem GPU, pule esta linha
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

## Protocolo experimental (evitando vazamento de partida)

O split oficial do repositório de referência é aleatório **por mensagem**, e
todas as 101 partidas do dataset aparecem ao mesmo tempo no treino e no teste.
Isso permite ao modelo "reconhecer" a partida (jogadores, contexto, estilo de
escrita) e infla as métricas.

Por isso o protocolo principal deste trabalho é uma **validação cruzada
estratificada e agrupada por partida** (`StratifiedGroupKFold`, k=5) sobre o
dataset completo (treino + teste oficiais):

- cada fold de teste contém partidas que o modelo nunca viu;
- a validação interna (escolha do melhor epoch) também é separada por partida;
- o particionamento é determinístico e fica em cache (`cache/folds_group_k5.csv`),
  então todos os modelos e o baseline usam exatamente os mesmos folds;
- reportamos média ± desvio padrão entre os folds e as métricas agregadas das
  predições out-of-fold (cada mensagem prevista uma única vez).

Para quantificar o efeito do vazamento, o mesmo procedimento é repetido com
folds aleatórios por mensagem (`--protocol message`), como no split oficial.

## Como rodar

```bash
# 1. Conferir os dados e os folds agrupados por partida (checa vazamento)
python src/data.py

# 2. Baseline zero-shot: unitary/toxic-bert, o melhor modelo pronto do paper
#    (reproduz o F1=0.59 do paper no teste oficial e avalia nos folds agrupados)
python src/baseline.py

# 3. Fine-tuning com validação cruzada agrupada por partida (protocolo principal)
python src/train.py --model distilbert --folds 5 --protocol group
python src/train.py --model bert       --folds 5 --protocol group
python src/train.py --model roberta    --folds 5 --protocol group

# 3b. Mesmo procedimento com folds por mensagem, para medir o vazamento
python src/train.py --model distilbert --folds 5 --protocol message

# 3c. Execução única no split oficial (gera o modelo usado pela demo)
python src/train.py --model distilbert

# 4. Avaliar e comparar os modelos treinados
python src/evaluate.py

# 4b. Comparação estatística nas predições out-of-fold (bootstrap pareado
#     por partida, subgrupos, limiar, exemplos de erro) -> outputs/comparison/
python src/compare.py

# 5. Análise própria: toxicidade por papel do campeão e relação com o ofensor
python src/analysis.py

# 6. Demo interativa
python src/demo.py --model distilbert
```

## Playground

Interface web para digitar mensagens e comparar os três modelos ajustados (e,
opcionalmente, o toxic-bert sem fine-tuning) lado a lado:

```bash
.venv\Scripts\streamlit run app/playground.py
```

Abre em http://localhost:8501. Precisa dos modelos em `outputs/<modelo>/best_model`
(gerados por `src/train.py --model <modelo>` ou baixados da pasta compartilhada).
No Colab, a seção 9 do notebook `colab/train_colab.ipynb` sobe a mesma
comparação com Gradio e gera um link público.

## Dataset e ground truth

Dataset: **L2DTnH**, mensagens de chat de League of Legends anotadas
manualmente como tóxicas/não-tóxicas por jogadores experientes (13.773
não-tóxicas / 1.398 tóxicas), extraído dos chatlogs do Tribunal de LoL.

Citação:
```bibtex
@inproceedings{ave2026botlane,
    author = {Ave, Jonas and Pekaric, Irdin and Frohner, Matthias and Apruzzese, Giovanni},
    title = {{"bot lane noob": Towards Practical Deployment of NLP-based Toxicity Detectors in Video Games}},
    booktitle = {European Symposium on Research in Computer Security (ESORICS)},
    year = {2026}
}
```

O script de fine-tuning original do repositório (`unitary/toxic-bert`, pipeline
manual) foi usado apenas como referência de formato dos dados; a implementação
em `src/train.py` é própria (Trainer API do HuggingFace, comparação entre
múltiplas arquiteturas) e a análise por papel de campeão em `src/analysis.py`
não existe no trabalho original.

## Declaração de uso de IA generativa

*(preencher antes da entrega, conforme exigido no item F do enunciado)*

- Ferramenta utilizada: Claude (Anthropic), via Claude Code.
- Etapas: brainstorming e validação do tópico/dataset; geração do esqueleto de
  código em `src/`; redação da proposta enviada no Moodle.
- Todo o código e texto gerado foi revisado e deve ser validado/testado pelos
  autores antes da entrega final; os resultados, análises e a redação do
  artigo são de responsabilidade dos autores.
