import os
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).parent / "CENG493_Project"
sys.path.insert(0, str(_PROJECT_DIR))

import json
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turkish Legal RAG — CENG493",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_RESULTS        = Path(__file__).parent / "results" / "stage_results"
_STAGE1_METRICS = _RESULTS / "stage1" / "baseline_metrics.json"
_ABLATION_JSON  = _RESULTS / "ablation_summary.json"

# ── Stage metadata ────────────────────────────────────────────────────────────
STAGE_ORDER  = ["stage1", "emb_ft", "llm_ft", "full"]
STAGE_LABELS = {
    "stage1": "Stage 1 — Baseline",
    "emb_ft": "Stage 2 — Emb. Fine-tuned",
    "llm_ft": "Stage 4 — LLM Fine-tuned",
    "full":   "Stage 5 — Full Optimized",
}

SAMPLE_QUESTIONS = [
    "Türk Medeni Kanunu'na göre evlilik için asgari yaş kaçtır?",
    "İş Kanunu'na göre haftalık normal çalışma süresi kaç saattir?",
    "Türk Ceza Kanunu'na göre kasten öldürme suçunun cezası nedir?",
    "Kira sözleşmesi hangi koşullarda sona erdirilebilir?",
    "Tüketici hakları kapsamında ayıplı mal iade süresi ne kadardır?",
    "Türk Borçlar Kanunu'na göre haksız fiil sorumluluğunun şartları nelerdir?",
]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics() -> dict:
    stages: dict = {}
    if _STAGE1_METRICS.exists():
        with open(_STAGE1_METRICS, encoding="utf-8") as f:
            stages["stage1"] = json.load(f)
    if _ABLATION_JSON.exists():
        with open(_ABLATION_JSON, encoding="utf-8") as f:
            ablation = json.load(f)
        for key in ("emb_ft", "llm_ft", "full"):
            if key in ablation:
                stages[key] = ablation[key]
    return stages


# ── Pipeline loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline (BGE-M3 + FAISS)…")
def load_pipeline():
    import config
    from retrieval.embedder import Embedder
    from retrieval.retriever import Retriever
    from generation.rag_pipeline import RAGPipeline

    embedder = Embedder()
    embedder.load_model()

    retriever = Retriever(
        embedder,
        index_path=config.INDEX_DIR / config.INDEX_FILE,
        metadata_path=config.INDEX_DIR / config.METADATA_FILE,
    )
    pipeline = RAGPipeline(retriever)
    return pipeline


def ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Turkish Legal RAG System")
st.caption(
    "CENG493 Term Project · Retrieval-Augmented Generation for Turkish Legal Question Answering"
)
st.divider()

tab_demo, tab_ablation, tab_arch = st.tabs(
    ["Live Demo", "Ablation Results", "Architecture"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Ask a Legal Question")

        sample = st.selectbox(
            "Sample questions:",
            ["— type your own —"] + SAMPLE_QUESTIONS,
        )
        default_text = sample if sample != "— type your own —" else ""

        question = st.text_area(
            "Question (Turkish)",
            value=default_text,
            height=110,
            placeholder="Türkçe hukuki sorunuzu buraya yazın…",
        )

        run = st.button("Get Answer", type="primary", use_container_width=True)

        ollama_ok = ollama_running()
        if not ollama_ok:
            st.warning(
                "Ollama is not running. Start it with `ollama serve` before querying.",
                icon="⚠️",
            )
        else:
            st.success("Ollama is online", icon="✅")

        st.caption(
            "The pipeline uses dense FAISS retrieval (BGE-M3) to find relevant legal "
            "passages, then generates an answer via Qwen 2.5 running locally."
        )

    with col_out:
        if run and question.strip():
            if not ollama_ok:
                st.error("Cannot generate — Ollama is offline.")
            else:
                with st.spinner("Retrieving passages and generating answer…"):
                    try:
                        pipeline = load_pipeline()
                        result = pipeline.run(question)

                        st.subheader("Generated Answer")
                        st.info(result["answer"])

                        st.subheader(f"Retrieved Passages ({len(result['retrieved_chunks'])})")
                        for i, chunk in enumerate(result["retrieved_chunks"], 1):
                            source  = chunk.get("source", "Unknown")
                            score   = chunk.get("score", 0.0)
                            text    = chunk.get("text", "")
                            preview = text[:600] + "…" if len(text) > 600 else text
                            with st.expander(f"{i}. {source}  —  score: {score:.4f}"):
                                st.markdown(f"```\n{preview}\n```")
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")
        elif run:
            st.warning("Please enter a question first.")
        else:
            st.markdown(
                """
                #### How it works

                1. Your question is embedded with **BGE-M3** (1024-dim)
                2. **FAISS** retrieves the top-10 most similar legal passages
                3. Passages are assembled into a context window (≤14,000 chars)
                4. **Qwen 2.5** generates a grounded answer with source citations

                Select a sample question or type your own, then click **Get Answer**.
                """
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABLATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ablation:
    stages = load_metrics()

    if not stages:
        st.error("Metrics files not found.")
        st.stop()

    present = [k for k in STAGE_ORDER if k in stages]
    xlabels = [STAGE_LABELS[k] for k in present]

    # ── Retrieval metrics ──────────────────────────────────────────────────────
    st.subheader("Retrieval Metrics")
    ret_metrics = {
        "Recall@5":  [stages[k]["retrieval_metrics"].get("recall_at_5",  0) for k in present],
        "Recall@10": [stages[k]["retrieval_metrics"].get("recall_at_10", 0) for k in present],
        "MRR":       [stages[k]["retrieval_metrics"].get("mrr",          0) for k in present],
        "nDCG@10":   [stages[k]["retrieval_metrics"].get("ndcg_at_10",   0) for k in present],
    }
    fig_ret = go.Figure()
    for metric, values in ret_metrics.items():
        fig_ret.add_trace(go.Bar(
            name=metric,
            x=xlabels,
            y=[round(v, 4) for v in values],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
    fig_ret.update_layout(
        barmode="group",
        height=380,
        yaxis=dict(range=[0, 1.15], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── QA metrics ────────────────────────────────────────────────────────────
    st.subheader("QA Metrics")
    qa_metrics = {
        "F1":               [stages[k]["qa_metrics"].get("f1",                    0) for k in present],
        "ROUGE-L":          [stages[k]["qa_metrics"].get("rouge_l",               0) for k in present],
        "BLEU":             [stages[k]["qa_metrics"].get("bleu",                  0) for k in present],
        "Citation Acc.":    [stages[k]["qa_metrics"].get("citation_accuracy",     0) for k in present],
        "Citation Present": [stages[k]["qa_metrics"].get("citation_presence_rate",0) for k in present],
    }
    fig_qa = go.Figure()
    for metric, values in qa_metrics.items():
        fig_qa.add_trace(go.Bar(
            name=metric,
            x=xlabels,
            y=[round(v, 4) for v in values],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
    fig_qa.update_layout(
        barmode="group",
        height=380,
        yaxis=dict(range=[0, 1.35], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig_qa, use_container_width=True)

    # ── Quality / hallucination metrics ───────────────────────────────────────
    st.subheader("Quality Metrics")
    qual_cols = st.columns(len(present))
    for col, k in zip(qual_cols, present):
        d     = stages[k]
        faith = d.get("faithfulness_rate") or d.get("hallucination_summary", {}).get("faithful_rate", 0)
        judge = d.get("llm_judge_score", 0)
        rel   = d.get("llm_relevancy_score", 0)
        coh   = d.get("llm_coherence_score", 0)
        sem   = d.get("semantic_similarity", 0)
        with col:
            st.markdown(f"**{STAGE_LABELS[k]}**")
            st.metric("Faithfulness", f"{faith:.1%}")
            st.metric("LLM Judge",    f"{judge:.3f}")
            st.metric("Relevancy",    f"{rel:.3f}")
            st.metric("Coherence",    f"{coh:.3f}")
            st.metric("Sem. Sim.",    f"{sem:.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Summary Table")
    rows = []
    for k in present:
        d         = stages[k]
        hp        = d.get("hyperparameters", {})
        rm        = d.get("retrieval_metrics", {})
        qm        = d.get("qa_metrics", {})
        emb_short = hp.get("embedding_model", "").replace("\\", "/").split("/")[-1]
        rows.append({
            "Stage":         STAGE_LABELS[k],
            "Embedding":     emb_short,
            "Retrieval":     hp.get("retrieval_mode", ""),
            "LLM":           hp.get("llm_model", ""),
            "Recall@5":      round(rm.get("recall_at_5", 0), 3),
            "MRR":           round(rm.get("mrr",          0), 3),
            "nDCG@10":       round(rm.get("ndcg_at_10",   0), 3),
            "F1":            round(qm.get("f1",            0), 3),
            "Citation Acc.": round(qm.get("citation_accuracy", 0), 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.subheader("Pipeline Flow")
        st.code(
            """\
User Query (Turkish)
        │
        ▼
┌───────────────────┐
│  BGE-M3 Embedder  │  1024-dim dense vector
└────────┬──────────┘
         │
         ▼
┌──────────────────────────┐
│    Hybrid Retrieval       │
│  ┌─────────┐ ┌─────────┐ │
│  │  FAISS  │ │  BM25   │ │  Top-10 candidates each
│  └────┬────┘ └────┬────┘ │
│       └─────┬─────┘      │
│         RRF Fusion        │
│              │            │
│   ┌──────────▼─────────┐  │
│   │  Cross-Encoder      │  │  BGE Reranker v2-m3
│   │  Reranker           │  │
│   └──────────┬─────────┘  │
└──────────────┼────────────┘
               │  Top-5 chunks
               ▼
      Context Assembly  ≤14,000 chars
               │
               ▼
┌──────────────────────────┐
│   Qwen 2.5 via Ollama    │  14B / LoRA fine-tuned
└──────────────┬───────────┘
               │
               ▼
      Answer + Source Citations""",
            language="text",
        )

    with col_b:
        st.subheader("5-Stage Ablation")
        st.markdown(
            """
| Stage | What was added |
|-------|----------------|
| **1 — Baseline** | BGE-M3 + RRF retrieval + Qwen 2.5-7B (no fine-tuning) |
| **2 — Emb. FT** | BGE-M3 contrastive fine-tuning on Turkish legal corpus |
| **3 — Reranker FT** | Cross-encoder fine-tuned for legal passage reranking |
| **4 — LLM FT** | Qwen 2.5 LoRA instruction-tuned on legal QA pairs |
| **5 — Full Optimized** | All three fine-tuned components combined |
"""
        )

        st.subheader("Dataset & Corpus")
        st.markdown(
            """
| | |
|--|--|
| **Corpus** | 12 Turkish laws (Anayasa, TMK, TCK, CMK, TBK, İK, …) |
| **Chunking** | 1,400 chars / 180 char overlap |
| **Index** | FAISS IndexFlatIP (~50k chunks) |
| **Eval set** | 300 questions — HMGS 2025 benchmark |
| **Metrics** | Recall@K, MRR, nDCG · F1, ROUGE-L, BLEU, Citation Acc. |
"""
        )

        st.subheader("Tech Stack")
        st.markdown(
            """
- **Embeddings** — `FlagEmbedding` BGE-M3 (dense + sparse + ColBERT)
- **Vector search** — `faiss-cpu` IndexFlatIP
- **Sparse retrieval** — `rank_bm25` + RRF fusion
- **Reranker** — `BAAI/bge-reranker-v2-m3` cross-encoder
- **LLM serving** — Qwen 2.5-14B via `Ollama` (local)
- **Fine-tuning** — `unsloth` LoRA / QLoRA
- **Evaluation** — `rouge_score`, `sacrebleu`, LLM-as-judge
"""
        )
