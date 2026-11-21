"""Playground: digite uma mensagem de chat e veja a avaliação dos três modelos.

Rodar a partir da raiz do projeto:
    .venv\\Scripts\\streamlit run app/playground.py
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import available_models, load_models  # noqa: E402
from riot_api import get_champion_map  # noqa: E402

EXAMPLES = [
    "gg wp",
    "uninstall the game you trash noob",
    "STUPID FUCKING LAG",
    "report jungle he is feeding",
    "bg",
    "l2p",
    "nice ult zed",
    "kys",
]

st.set_page_config(page_title="Toxicidade no chat de LoL", page_icon="🛡️", layout="wide")

# Medidor: preenchimento azul sobre trilho do mesmo tom (paleta sequencial),
# marca vertical no limiar; o veredito vem como ícone + texto, nunca só cor.
st.markdown("""
<style>
.meter-list { display: grid; gap: 14px; margin: 8px 0 4px; max-width: 900px; }
.meter-row { display: grid; grid-template-columns: minmax(150px, 1.2fr) 3fr 64px minmax(120px, 1fr);
             align-items: center; gap: 12px; }
.meter-name { font-weight: 600; line-height: 1.2; }
.meter-name small { display: block; font-weight: 400; opacity: .65; font-size: .78rem; }
.meter-track { position: relative; height: 10px; border-radius: 5px; background: #cde2fb; }
.meter-fill { height: 100%; border-radius: 5px; background: #2a78d6; min-width: 2px; }
.meter-tick { position: absolute; top: -4px; width: 2px; height: 18px; background: currentColor; opacity: .55; }
.meter-value { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.verdict { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.verdict .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
@media (prefers-color-scheme: dark) {
  .meter-track { background: #0d366b; }
  .meter-fill { background: #3987e5; }
}
@media (max-width: 640px) {
  .meter-row { grid-template-columns: 1fr 64px; }
  .meter-track { grid-column: 1 / -1; grid-row: 2; }
  .verdict { grid-column: 1 / -1; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Carregando modelos...")
def get_models(include_baseline):
    return load_models(include_baseline=include_baseline, device="cpu")


@st.cache_resource
def get_champions():
    try:
        return get_champion_map()
    except Exception:
        return {}


def verdict_html(is_toxic):
    color, icon, label = ("#d03b3b", "⚠", "Tóxica") if is_toxic else ("#0ca30c", "✓", "Não tóxica")
    return f'<span class="verdict"><span class="dot" style="background:{color}"></span>{icon} {label}</span>'


def meter_list_html(rows):
    out = ['<div class="meter-list">']
    for r in rows:
        pct = 100 * r["prob"]
        f1 = f"F1 na CV: {r['cv_f1']:.3f}" if r["cv_f1"] is not None else ""
        out.append(
            f'<div class="meter-row" title="{html.escape(r["name"])}: {pct:.1f}% de probabilidade de ser tóxica">'
            f'<div class="meter-name">{html.escape(r["name"])}<small>{f1}</small></div>'
            f'<div class="meter-track"><div class="meter-fill" style="width:{pct:.1f}%"></div>'
            f'<div class="meter-tick" style="left:calc({100 * r["threshold"]:.1f}% - 1px)"></div></div>'
            f'<div class="meter-value">{pct:.1f}%</div>'
            f'<div>{verdict_html(r["toxic"])}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def find_champion_mentions(text, champion_map):
    lowered = text.lower()
    return [n for n in champion_map
            if re.search(rf"(?<![a-z0-9]){re.escape(n.lower())}(?![a-z0-9])", lowered)]


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("Configurações")
    threshold = st.slider("Limiar de decisão dos modelos ajustados", 0.05, 0.95, 0.50, 0.05,
                          help="Probabilidade a partir da qual a mensagem é considerada tóxica. "
                               "Os modelos foram avaliados com 0,5.")
    include_baseline = st.toggle("Comparar com o toxic-bert (sem fine-tuning)", value=True,
                                 help="Melhor modelo pronto avaliado no paper de referência; usa limiar fixo de 0,45.")
    st.divider()
    st.caption("Modelos ajustados no dataset L2DTnH (chat do Tribunal do LoL). "
               "O F1 mostrado é o da validação cruzada agrupada por partida.")

if not available_models():
    st.error("Nenhum modelo encontrado em `outputs/<modelo>/best_model`. Treine com `src/train.py` "
             "ou baixe a pasta de modelos do Drive.")
    st.stop()

models = get_models(include_baseline)


def evaluate(texts):
    table = pd.DataFrame({"mensagem": texts})
    for m in models:
        t = m.default_threshold if m.baseline else threshold
        table[m.name] = m.predict_proba(texts)
        table[f"{m.name} tóxica"] = table[m.name] >= t
    return table


st.title("🛡️ Toxicidade no chat de LoL")
st.write("Digite uma mensagem de chat em inglês e veja como cada modelo a classifica.")

tab_one, tab_many = st.tabs(["Uma mensagem", "Várias mensagens"])

# ------------------------------------------------------------------ uma mensagem
with tab_one:
    st.caption("Exemplos:")
    picked = st.pills("Exemplos", EXAMPLES, label_visibility="collapsed", key="example")
    text = st.text_input("Mensagem", value=picked or "", placeholder="ex.: report mid, he is trolling",
                         key=f"msg_{picked}")

    if text.strip():
        row = evaluate([text.strip()]).iloc[0]
        rows = [{"name": m.name, "prob": row[m.name], "toxic": bool(row[f"{m.name} tóxica"]),
                 "threshold": m.default_threshold if m.baseline else threshold, "cv_f1": m.cv_f1}
                for m in models]
        tuned = [r for r in rows if "sem fine-tuning" not in r["name"]]
        n_toxic = sum(r["toxic"] for r in tuned)
        if n_toxic in (0, len(tuned)):
            st.markdown(f"**Os {len(tuned)} modelos ajustados concordam:** "
                        + verdict_html(n_toxic == len(tuned)), unsafe_allow_html=True)
        else:
            st.markdown(f"**Os modelos ajustados discordam:** {n_toxic} de {len(tuned)} consideram tóxica.")

        st.markdown(meter_list_html(rows), unsafe_allow_html=True)
        st.caption("A barra é a probabilidade de a mensagem ser tóxica; o traço vertical marca o limiar de decisão.")

        mentions = find_champion_mentions(text, get_champions())
        if mentions:
            champs = get_champions()
            st.markdown("**Campeões mencionados** (Data Dragon): " + "; ".join(
                f"{n} ({', '.join(champs[n]['tags'])})" for n in mentions))

        with st.expander("Ver como tabela"):
            st.dataframe(pd.DataFrame([{"modelo": r["name"], "probabilidade tóxica": round(r["prob"], 4),
                                        "limiar": r["threshold"], "tóxica": "sim" if r["toxic"] else "não"}
                                       for r in rows]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ várias mensagens
with tab_many:
    batch = st.text_area("Uma mensagem por linha", height=180,
                         value="gg wp\nreport jungle he is feeding\nyou are so bad uninstall\nSTUPID FUCKING LAG\nbg")
    texts = [t.strip() for t in batch.splitlines() if t.strip()]
    if texts:
        table = evaluate(texts)
        tuned = [m for m in models if not m.baseline]
        votes = table[[f"{m.name} tóxica" for m in tuned]].sum(axis=1)
        table["ajustados"] = [f"⚠ {v} de {len(tuned)} tóxica" if v else f"✓ 0 de {len(tuned)} tóxica" for v in votes]
        short = {m.name: m.name.replace(" (sem fine-tuning)", "") for m in models}
        cfg = {short[m.name]: st.column_config.ProgressColumn(
                   short[m.name], min_value=0.0, max_value=1.0, format="percent",
                   help=f"Probabilidade de ser tóxica (limiar {m.default_threshold if m.baseline else threshold:.2f})")
               for m in models}
        view = table[["mensagem"] + [m.name for m in models] + ["ajustados"]].rename(columns=short)
        st.dataframe(view, column_config=cfg, hide_index=True, use_container_width=True)
        st.caption(f"Barras: probabilidade de ser tóxica. \"ajustados\": quantos dos {len(tuned)} modelos ajustados "
                   f"passam do limiar de {threshold:.2f}.")
        st.download_button("Baixar CSV completo", table.to_csv(index=False).encode("utf-8"),
                           "avaliacao_toxicidade.csv", "text/csv")
