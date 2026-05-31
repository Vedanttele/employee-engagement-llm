"""
streamlit_app.py — Interactive dashboard for employee engagement survey analysis.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Engagement Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import project modules ─────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    SENTIMENT_COLORS, ENGAGEMENT_THEMES, SUPPORTED_LANGUAGES,
    PROCESSED_DIR, SYNTHETIC_DIR
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        font-weight: 400;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .sub-header {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #6b7280;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8f9fe 0%, #eef0ff 100%);
        border: 1px solid #e0e3ff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #1a1a2e;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
    }

    .response-card {
        background: #fafafa;
        border-left: 4px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .tag {
        display: inline-block;
        background: #eef0ff;
        color: #4338ca;
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-size: 0.72rem;
        font-family: 'DM Mono', monospace;
        margin: 0.1rem;
    }

    .urgent-badge {
        background: #fee2e2;
        color: #b91c1c;
        border-radius: 4px;
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        font-weight: 600;
    }

    div[data-testid="stSelectbox"] > div {
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the best available dataset."""
    candidates = [
        PROCESSED_DIR / "topics.csv",
        PROCESSED_DIR / "analyzed_responses.csv",
        PROCESSED_DIR / "cleaned_responses.csv",
        SYNTHETIC_DIR / "survey_responses.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Normalize column names for backward compatibility
            if "llm_sentiment" not in df and "sentiment_label" in df:
                df["llm_sentiment"] = df["sentiment_label"]
            if "llm_summary_en" not in df:
                df["llm_summary_en"] = df.get("response_text", "")
            if "llm_action_signal" not in df:
                df["llm_action_signal"] = "no_action"
            if "engagement_score" not in df:
                score_map = {"very_positive": 100, "positive": 75,
                             "neutral": 50, "negative": 25, "very_negative": 0}
                df["engagement_score"] = df["llm_sentiment"].map(score_map).fillna(50)
            return df
    return _generate_demo_data()


def _generate_demo_data() -> pd.DataFrame:
    """Generate small demo dataset when no real data exists."""
    import random, numpy as np
    random.seed(42)
    np.random.seed(42)

    langs = list(SUPPORTED_LANGUAGES.keys())
    themes = list(ENGAGEMENT_THEMES.keys())
    sentiments = ["very_positive", "positive", "neutral", "negative", "very_negative"]
    departments = ["Engineering", "Product", "Finance", "HR", "Operations", "Sales"]

    sample_texts = {
        "en": ["The team culture here is amazing, I feel genuinely supported.",
               "Work-life balance has been difficult lately with constant deadlines.",
               "Management communication could be much more transparent.",
               "I appreciate the growth opportunities provided by the company.",
               "Salary benchmarking feels outdated compared to market standards."],
        "de": ["Die Unternehmenskultur hier ist wirklich inspirierend.",
               "Ich fühle mich manchmal überwältigt von der Arbeitsbelastung.",
               "Die Führung könnte transparenter kommunizieren.",
               "Tolles Team, ich fühle mich wohl und wertgeschätzt.",
               "Die Gehaltsstruktur entspricht nicht mehr dem Markt."],
        "fr": ["L'ambiance d'équipe est excellente et motivante.",
               "L'équilibre travail-vie personnelle laisse à désirer.",
               "La direction manque de transparence dans ses décisions.",
               "Je me sens reconnu et valorisé dans mon rôle.",
               "Les outils mis à disposition sont parfois obsolètes."],
        "es": ["El ambiente laboral es positivo y colaborativo.",
               "La carga de trabajo ha aumentado sin reconocimiento adicional.",
               "Me gustaría más comunicación clara de la dirección.",
               "Las oportunidades de desarrollo profesional son excelentes.",
               "La compensación no refleja el mercado actual."],
        "hi": ["यहाँ की टीम संस्कृति बहुत अच्छी है।",
               "काम का बोझ कभी-कभी बहुत ज़्यादा हो जाता है।",
               "प्रबंधन अधिक पारदर्शी हो सकता है।",
               "करियर विकास के अवसर बेहतरीन हैं।",
               "वेतन संरचना बाज़ार के अनुरूप नहीं है।"],
        "zh": ["团队文化非常出色，我感到被充分支持。",
               "最近工作与生活的平衡变得很困难。",
               "管理层的沟通可以更加透明。",
               "公司提供的发展机会非常好。",
               "薪酬结构似乎落后于市场标准。"],
    }

    rows = []
    sentiment_dist = [0.1, 0.3, 0.25, 0.25, 0.1]
    for _ in range(300):
        lang = random.choice(langs)
        sentiment = random.choices(sentiments, weights=sentiment_dist)[0]
        text = random.choice(sample_texts.get(lang, sample_texts["en"]))
        rows.append({
            "response_text": text,
            "language_code": lang,
            "language_name": SUPPORTED_LANGUAGES[lang],
            "theme_key": random.choice(themes),
            "theme_display": ENGAGEMENT_THEMES[random.choice(themes)],
            "llm_sentiment": sentiment,
            "llm_primary_theme": random.choice(themes),
            "llm_summary_en": text[:80] + "..." if len(text) > 80 else text,
            "llm_action_signal": random.choice(["no_action", "monitor", "positive_share", "urgent_action"]),
            "llm_key_phrases": json.dumps([text.split()[0], text.split()[-1]] if text.split() else []),
            "llm_emotion_tags": json.dumps(random.choice([["satisfied"], ["frustrated"], ["hopeful"], ["neutral"]])),
            "department": random.choice(departments),
            "engagement_score": {"very_positive": 100, "positive": 75, "neutral": 50,
                                 "negative": 25, "very_negative": 0}[sentiment],
            "data_source": "demo",
        })
    return pd.DataFrame(rows)


# ── Header ─────────────────────────────────────────────────────────────────────
df_full = load_data()

st.markdown('<div class="main-header">Employee Engagement Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multilingual Survey Analysis · Powered by Claude AI</div>', unsafe_allow_html=True)

# ── Sidebar Filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")

    available_langs = sorted(df_full["language_code"].unique())
    lang_options = {SUPPORTED_LANGUAGES.get(l, l): l for l in available_langs}
    selected_lang_names = st.multiselect(
        "Language",
        options=list(lang_options.keys()),
        default=list(lang_options.keys()),
    )
    selected_langs = [lang_options[n] for n in selected_lang_names]

    available_sentiments = ["very_positive", "positive", "neutral", "negative", "very_negative"]
    selected_sentiments = st.multiselect(
        "Sentiment",
        options=available_sentiments,
        default=available_sentiments,
    )

    available_themes = sorted(df_full.get("llm_primary_theme", df_full.get("theme_key", pd.Series())).unique())
    theme_display_map = {k: v for k, v in ENGAGEMENT_THEMES.items()}
    theme_options = {theme_display_map.get(t, t): t for t in available_themes}
    selected_theme_names = st.multiselect(
        "Theme",
        options=list(theme_options.keys()),
        default=list(theme_options.keys()),
    )
    selected_themes = [theme_options[n] for n in selected_theme_names]

    if "department" in df_full.columns:
        departments = ["All"] + sorted(df_full["department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department", departments)
    else:
        selected_dept = "All"

    st.markdown("---")
    st.markdown("### ⚡ Action Signals")
    show_urgent = st.checkbox("Urgent Action Only", False)

# ── Apply filters ──────────────────────────────────────────────────────────────
df = df_full.copy()
if selected_langs:
    df = df[df["language_code"].isin(selected_langs)]
if selected_sentiments:
    df = df[df["llm_sentiment"].isin(selected_sentiments)]

theme_col = "llm_primary_theme" if "llm_primary_theme" in df.columns else "theme_key"
if selected_themes:
    df = df[df[theme_col].isin(selected_themes)]
if selected_dept != "All" and "department" in df.columns:
    df = df[df["department"] == selected_dept]
if show_urgent:
    df = df[df.get("llm_action_signal", "no_action") == "urgent_action"]

if len(df) == 0:
    st.warning("No responses match the current filters. Adjust the sidebar.")
    st.stop()

# ── KPI Row ────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

avg_score = df["engagement_score"].mean() if "engagement_score" in df.columns else 0
positive_pct = (df["llm_sentiment"].isin(["positive", "very_positive"])).mean() * 100
urgent_count = (df.get("llm_action_signal", pd.Series(["no_action"] * len(df))) == "urgent_action").sum()
languages_count = df["language_code"].nunique()
themes_count = df[theme_col].nunique()

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_score:.0f}</div>
        <div class="metric-label">Engagement Score</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{positive_pct:.0f}%</div>
        <div class="metric-label">Positive Rate</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#b91c1c">{urgent_count}</div>
        <div class="metric-label">Urgent Signals</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Responses</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{languages_count}</div>
        <div class="metric-label">Languages</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🌍 Language View", "🎯 Theme Analysis",
    "⚡ Action Items", "🔍 Explorer"
])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Sentiment Distribution")
        sent_counts = df["llm_sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        # Preserve order
        order = ["very_positive", "positive", "neutral", "negative", "very_negative"]
        sent_counts["Sentiment"] = pd.Categorical(sent_counts["Sentiment"], categories=order, ordered=True)
        sent_counts = sent_counts.sort_values("Sentiment")
        colors = [SENTIMENT_COLORS.get(s, "#aaa") for s in sent_counts["Sentiment"]]

        fig = px.bar(
            sent_counts, x="Sentiment", y="Count",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Engagement Score by Theme")
        theme_scores = df.groupby(theme_col)["engagement_score"].mean().reset_index()
        theme_scores.columns = ["Theme", "Score"]
        theme_scores["Theme"] = theme_scores["Theme"].map(
            lambda x: ENGAGEMENT_THEMES.get(x, x)[:30]
        )
        theme_scores = theme_scores.sort_values("Score", ascending=True)

        fig2 = px.bar(
            theme_scores, x="Score", y="Theme", orientation="h",
            color="Score", color_continuous_scale=["#E74C3C", "#F39C12", "#2ECC71"],
            range_color=[0, 100],
        )
        fig2.update_layout(showlegend=False, height=300,
                           coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=10, b=0),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Sentiment heatmap: Language x Theme
    st.markdown("#### Sentiment Heatmap — Language × Theme")
    pivot = df.groupby(["language_code", theme_col])["engagement_score"].mean().unstack(fill_value=50)
    pivot.index = [SUPPORTED_LANGUAGES.get(l, l) for l in pivot.index]
    pivot.columns = [ENGAGEMENT_THEMES.get(c, c)[:20] for c in pivot.columns]

    fig3 = px.imshow(
        pivot, color_continuous_scale=["#E74C3C", "#F7DC6F", "#2ECC71"],
        zmin=0, zmax=100, aspect="auto",
        labels=dict(color="Engagement Score"),
    )
    fig3.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 2: Language View ───────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Responses by Language")
    lang_counts = df.groupby("language_code").agg(
        Count=("response_text", "count"),
        Avg_Score=("engagement_score", "mean"),
    ).reset_index()
    lang_counts["Language"] = lang_counts["language_code"].map(SUPPORTED_LANGUAGES)
    lang_counts["Avg_Score"] = lang_counts["Avg_Score"].round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(lang_counts, values="Count", names="Language",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(lang_counts.sort_values("Avg_Score"), x="Avg_Score", y="Language",
                      orientation="h", color="Avg_Score",
                      color_continuous_scale=["#E74C3C", "#2ECC71"],
                      range_color=[0, 100])
        fig2.update_layout(height=300, showlegend=False, coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=10, b=0),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Sentiment per Language")
    lang_sent = df.groupby(["language_code", "llm_sentiment"]).size().reset_index(name="Count")
    lang_sent["Language"] = lang_sent["language_code"].map(SUPPORTED_LANGUAGES)
    fig3 = px.bar(lang_sent, x="Language", y="Count", color="llm_sentiment",
                  color_discrete_map=SENTIMENT_COLORS, barmode="stack")
    fig3.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 3: Theme Analysis ──────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Theme Distribution")
    theme_counts = df[theme_col].value_counts().reset_index()
    theme_counts.columns = ["Theme_Key", "Count"]
    theme_counts["Theme"] = theme_counts["Theme_Key"].map(
        lambda x: ENGAGEMENT_THEMES.get(x, x)
    )

    fig = px.treemap(theme_counts, path=["Theme"], values="Count",
                     color="Count", color_continuous_scale="Blues")
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Theme × Sentiment Breakdown")
    ts = df.groupby([theme_col, "llm_sentiment"]).size().reset_index(name="Count")
    ts["Theme"] = ts[theme_col].map(lambda x: ENGAGEMENT_THEMES.get(x, x)[:25])
    fig2 = px.bar(ts, x="Theme", y="Count", color="llm_sentiment",
                  color_discrete_map=SENTIMENT_COLORS, barmode="stack")
    fig2.update_xaxes(tickangle=30)
    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=60),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 4: Action Items ────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 🚨 Urgent Action Required")
    if "llm_action_signal" in df.columns:
        urgent = df[df["llm_action_signal"] == "urgent_action"]
        if len(urgent) > 0:
            for _, row in urgent.head(10).iterrows():
                lang = SUPPORTED_LANGUAGES.get(row["language_code"], row["language_code"])
                theme = ENGAGEMENT_THEMES.get(row.get(theme_col, ""), "Unknown Theme")
                st.markdown(f"""
                <div class="response-card" style="border-left-color: #E74C3C;">
                    <span class="urgent-badge">URGENT</span>&nbsp;
                    <span class="tag">{lang}</span>
                    <span class="tag">{theme[:25]}</span>
                    <p style="margin-top: 0.5rem; color: #374151;">{row['response_text']}</p>
                    <em style="color:#6b7280; font-size:0.8rem;">
                        Summary: {row.get('llm_summary_en', '')}
                    </em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No urgent action signals in the current filter.")

    st.markdown("#### ⭐ Positive Highlights to Share")
    if "llm_action_signal" in df.columns:
        pos_share = df[df["llm_action_signal"] == "positive_share"]
        for _, row in pos_share.head(5).iterrows():
            lang = SUPPORTED_LANGUAGES.get(row["language_code"], row["language_code"])
            theme = ENGAGEMENT_THEMES.get(row.get(theme_col, ""), "Unknown Theme")
            st.markdown(f"""
            <div class="response-card" style="border-left-color: #2ECC71;">
                <span class="tag">{lang}</span>
                <span class="tag">{theme[:25]}</span>
                <p style="margin-top: 0.5rem; color: #374151;">{row['response_text']}</p>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 5: Explorer ────────────────────────────────────────────────────────────
with tab5:
    st.markdown("#### Browse Individual Responses")
    sort_opts = {"Newest First": False, "Lowest Score": True, "Highest Score": False}
    sort_by_score = st.radio("Sort by", ["Lowest Score", "Highest Score"], horizontal=True)

    df_explorer = df.sort_values("engagement_score", ascending=(sort_by_score == "Lowest Score"))

    search = st.text_input("🔍 Search in responses", placeholder="e.g. management, Gehalt, équipe")
    if search:
        df_explorer = df_explorer[
            df_explorer["response_text"].str.contains(search, case=False, na=False)
        ]

    st.markdown(f"Showing {min(50, len(df_explorer))} of {len(df_explorer)} responses")

    for _, row in df_explorer.head(50).iterrows():
        lang = SUPPORTED_LANGUAGES.get(row["language_code"], row["language_code"])
        theme = ENGAGEMENT_THEMES.get(row.get(theme_col, ""), "Unknown")
        sentiment = row.get("llm_sentiment", "neutral")
        score = row.get("engagement_score", 50)
        color = SENTIMENT_COLORS.get(sentiment, "#aaa")

        emotions = []
        if "llm_emotion_tags" in row and pd.notna(row["llm_emotion_tags"]):
            try:
                emotions = json.loads(row["llm_emotion_tags"])
            except Exception:
                pass

        emotion_tags = " ".join([f'<span class="tag">{e}</span>' for e in emotions])

        st.markdown(f"""
        <div class="response-card" style="border-left-color: {color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="tag">{lang}</span>
                    <span class="tag">{theme[:22]}</span>
                    <span class="tag" style="background:{color}22; color:{color};">{sentiment.replace('_',' ')}</span>
                    {emotion_tags}
                </div>
                <div style="font-family:'DM Serif Display'; font-size:1.4rem; color:{color};">
                    {score:.0f}
                </div>
            </div>
            <p style="margin: 0.6rem 0 0.2rem; color:#1f2937; line-height:1.5;">{row['response_text']}</p>
            <em style="color:#9ca3af; font-size:0.78rem;">
                {row.get('llm_summary_en', '')[:100]}
            </em>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#9ca3af; font-size:0.75rem; font-family:DM Mono,monospace;">'
    'Employee Engagement Intelligence · Built with Claude AI · Multilingual NLP Pipeline'
    '</div>',
    unsafe_allow_html=True
)
