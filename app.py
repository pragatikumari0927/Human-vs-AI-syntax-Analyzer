import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from utils import FEATURE_LABELS, FEATURES, analyze_text, clean_text, load_nlp

st.set_page_config(page_title="Human vs AI Syntax Analyzer", page_icon="🔤", layout="wide")

st.title("🔤 Human vs AI Conversational Syntax Analyzer")
st.caption("Corpus dashboard + live syntactic analysis, built on spaCy dependency parsing.")

tab_dashboard, tab_live = st.tabs(["📊 Corpus Dashboard", "✍️ Live Analysis"])


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(
        ((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2)
    )
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan


# =================================================================
# TAB 1 — CORPUS DASHBOARD
# =================================================================
with tab_dashboard:
    st.subheader("Load your corpus results")
    st.write(
        "Upload the CSVs exported from `Human_vs_AI_Syntax_Corpus.ipynb` (Section 9). "
        "`human_ai_syntactic_features.csv` is required; "
        "`human_ai_statistical_comparison.csv` is optional — if you skip it, "
        "the app computes the same stats on the fly."
    )

    col1, col2 = st.columns(2)
    with col1:
        feat_file = st.file_uploader("human_ai_syntactic_features.csv", type="csv", key="feat")
    with col2:
        stats_file = st.file_uploader(
            "human_ai_statistical_comparison.csv (optional)", type="csv", key="stats"
        )

    if feat_file is not None:
        df_feat = pd.read_csv(feat_file)
        st.session_state["df_feat"] = df_feat

        required_cols = {"speaker"} | set(FEATURES)
        missing = required_cols - set(df_feat.columns)

        if missing:
            st.error(f"Missing expected columns: {sorted(missing)}")
        else:
            st.success(f"Loaded {len(df_feat):,} annotated sentences.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total sentences", f"{len(df_feat):,}")
            m2.metric("Human sentences", f"{(df_feat['speaker'] == 'human').sum():,}")
            m3.metric("AI sentences", f"{(df_feat['speaker'] == 'ai').sum():,}")

            st.markdown("### Descriptive statistics")
            st.dataframe(
                df_feat.groupby("speaker")[FEATURES].agg(["mean", "std"]).round(3),
                use_container_width=True,
            )

            st.markdown("### Statistical comparison")
            if stats_file is not None:
                df_results = pd.read_csv(stats_file)
                st.dataframe(df_results, use_container_width=True)
            else:
                human = df_feat[df_feat["speaker"] == "human"]
                ai = df_feat[df_feat["speaker"] == "ai"]
                rows = []
                for feat in FEATURES:
                    u_stat, p_val = stats.mannwhitneyu(
                        human[feat], ai[feat], alternative="two-sided"
                    )
                    d = cohens_d(human[feat], ai[feat])
                    rows.append(
                        {
                            "feature": feat,
                            "human_mean": human[feat].mean(),
                            "ai_mean": ai[feat].mean(),
                            "p_value": p_val,
                            "significant (p<0.05)": p_val < 0.05,
                            "cohens_d": d,
                        }
                    )
                df_results = pd.DataFrame(rows).round(4)
                st.dataframe(df_results, use_container_width=True)
                st.caption(
                    "Computed on the fly (Mann-Whitney U + Cohen's d) since no stats CSV was uploaded."
                )

            st.markdown("### Visualizations")
            chosen = st.selectbox(
                "Feature to plot", FEATURES, format_func=lambda f: FEATURE_LABELS[f]
            )
            fig = px.box(
                df_feat,
                x="speaker",
                y=chosen,
                color="speaker",
                title=f"{FEATURE_LABELS[chosen]}: human vs AI",
            )
            st.plotly_chart(fig, use_container_width=True)

            pos_feats = ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pron_ratio"]
            pos_means = (
                df_feat.groupby("speaker")[pos_feats]
                .mean()
                .reset_index()
                .melt(id_vars="speaker", var_name="pos", value_name="mean_ratio")
            )
            fig2 = px.bar(
                pos_means,
                x="pos",
                y="mean_ratio",
                color="speaker",
                barmode="group",
                title="Mean POS ratio per sentence: human vs AI",
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Upload `human_ai_syntactic_features.csv` to populate the dashboard.")


# =================================================================
# TAB 2 — LIVE ANALYSIS
# =================================================================
with tab_live:
    st.subheader("Analyze any text live")
    st.write(
        "Paste a passage below. It's split into sentences and each one is parsed with "
        "spaCy to extract the same syntactic features used in the corpus study."
    )

    label = st.radio("Tag this text as:", ["human", "ai", "unlabeled"], horizontal=True)
    text_input = st.text_area(
        "Text to analyze", height=180, placeholder="Paste a paragraph of human or AI writing here..."
    )

    if st.button("Analyze", type="primary"):
        if not text_input.strip():
            st.warning("Paste some text first.")
        else:
            nlp = load_nlp()
            df_live = analyze_text(nlp, clean_text(text_input), label=label)

            if df_live.empty:
                st.warning("No sentences with 3+ words were found — try a longer passage.")
            else:
                st.success(f"Parsed {len(df_live)} sentence(s).")
                st.dataframe(df_live[["sentence"] + FEATURES], use_container_width=True)

                st.markdown("#### Aggregate features for this text")
                agg = df_live[FEATURES].mean().rename(FEATURE_LABELS).round(3)
                st.dataframe(agg.to_frame("mean value"), use_container_width=True)

                if "df_feat" in st.session_state:
                    st.markdown("#### How this compares to your corpus")
                    corpus = st.session_state["df_feat"]
                    corpus_means = corpus.groupby("speaker")[FEATURES].mean()
                    compare = pd.DataFrame(
                        {
                            "this text": df_live[FEATURES].mean(),
                            "corpus human mean": corpus_means.loc["human"]
                            if "human" in corpus_means.index
                            else np.nan,
                            "corpus AI mean": corpus_means.loc["ai"]
                            if "ai" in corpus_means.index
                            else np.nan,
                        }
                    ).round(3)
                    compare.index = compare.index.map(FEATURE_LABELS)
                    st.dataframe(compare, use_container_width=True)
                else:
                    st.caption(
                        "Load a corpus in the Dashboard tab to compare this text against "
                        "human/AI corpus averages."
                    )
