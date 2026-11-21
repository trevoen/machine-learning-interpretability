# Detecção de toxicidade no chat de League of Legends

Trabalho final de PLN — detecção de discurso de ódio/toxicidade em mensagens de
chat de partidas de League of Legends, com fine-tuning de Transformers e uma
análise original cruzando toxicidade com o arquétipo do campeão mencionado
(via Data Dragon, API pública da Riot Games).

## Estrutura

```
src/
  config.py     caminhos e hiperparâmetros
  data.py       carregamento/limpeza dos splits treino/teste
  riot_api.py   integração com a Data Dragon (dados estáticos de campeões)
  train.py      fine-tuning comparativo (DistilBERT / BERT / RoBERTa)
  evaluate.py   avaliação e tabela comparativa entre modelos
  analysis.py   contribuição própria: toxicidade x papel do campeão (qui-quadrado)
  demo.py       demo interativa de linha de comando
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
pip install -r requirements.txt
```

## Como rodar

```bash
# 1. Conferir se os splits carregam corretamente
python src/data.py

# 2. Treinar um modelo (repita trocando --model para comparar)
python src/train.py --model distilbert
python src/train.py --model bert
python src/train.py --model roberta

# 3. Avaliar e comparar todos os modelos treinados
python src/evaluate.py

# 4. Análise própria: toxicidade por papel do campeão
python src/analysis.py

# 5. Demo interativa
python src/demo.py --model distilbert
```

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
