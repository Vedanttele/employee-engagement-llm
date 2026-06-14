"""
Streamlit dashboard for Employee Engagement Intelligence.

Connects to the FastAPI backend (default: http://localhost:8000).
Supports: file upload, real-time analysis, insights generation, export.
"""

import json
import uuid
import streamlit as st
import pandas as pd
import plotly.express as px
import httpx

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Engagement Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = st.secrets.get("API_BASE", "http://localhost:8000/api/v1")

SENTIMENT_COLORS = {
    "very_positive": "#2ECC71",
    "positive": "#82E0AA",
    "neutral": "#AEB6BF",
    "negative": "#E59866",
    "very_negative": "#E74C3C",
}

BURNOUT_COLORS = {
    "low": "#2ECC71",
    "medium": "#F1C40F",
    "high": "#E67E22",
    "critical": "#E74C3C",
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-header { font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: #1a1a2e; }
    .sub-header { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #6b7280;
                  letter-spacing: 0.1em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def call_analyze(records: list[dict], run_topics: bool = False) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_BASE}/survey/analyze",
            json={"responses": records, "run_topic_modeling": run_topics},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")
    return None


def call_insights(analyzed: list[dict]) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_BASE}/insights/generate",
            json={"analyzed_responses": analyzed},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Insights error: {exc}")
    return None


def ensure_response_id(records: list[dict]) -> list[dict]:
    for i, r in enumerate(records):
        if not r.get("response_id"):
            r["response_id"] = f"resp_{i:05d}_{uuid.uuid4().hex[:6]}"
    return records


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Employee Engagement Intelligence")
    st.caption("Powered by Claude AI")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Analyze Surveys", "Insights Report", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    run_topics = st.toggle("Topic Modeling", value=False, help="Requires >=10 responses")


# ── Page: Analyze Surveys ──────────────────────────────────────────────────────

if page == "Analyze Surveys":
    st.markdown('<p class="main-header">Survey Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">PII Anonymization · Sentiment · Burnout Risk · Topics</p>',
        unsafe_allow_html=True,
    )

    tab_upload, tab_paste, tab_sample = st.tabs(["Upload CSV/JSON", "Paste JSON", "Sample Data"])
    records: list[dict] = []

    with tab_upload:
        uploaded = st.file_uploader("Upload survey responses", type=["csv", "json"])
        if uploaded:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
                records = df_raw.rename(
                    columns={"id": "response_id", "response": "text", "lang": "language"}
                ).to_dict(orient="records")
            else:
                records = json.load(uploaded)
            st.success(f"Loaded {len(records)} records")

    with tab_paste:
        raw_json = st.text_area("Paste JSON array of survey responses", height=200)
        if raw_json.strip():
            try:
                records = json.loads(raw_json)
                st.success(f"Parsed {len(records)} records")
            except json.JSONDecodeError as e:
                st.error(f"JSON parse error: {e}")

    with tab_sample:
        if st.button("Load Sample (5 responses)"):
            records = [
                {"response_id": "s1", "text": "I love working here, great team culture!", "department": "Engineering", "language": "en"},
                {"response_id": "s2", "text": "The workload is unsustainable. I haven't had a day off in months.", "department": "Operations", "language": "en"},
                {"response_id": "s3", "text": "My manager never gives feedback. I don't know if I'm doing well.", "department": "Sales", "language": "en"},
                {"response_id": "s4", "text": "Tolle Teamarbeit, aber die Work-Life-Balance könnte besser sein.", "department": "HR", "language": "de"},
                {"response_id": "s5", "text": "Career growth opportunities are excellent. I feel valued and challenged.", "department": "Product", "language": "en"},
            ]
            st.success("Sample data loaded")

    if records:
        records = ensure_response_id(records)
        st.dataframe(pd.DataFrame(records).head(10), use_container_width=True)
        st.metric("Records ready", len(records))

        if st.button("Analyze", type="primary", use_container_width=True):
            with st.spinner("Running pipeline: Validate → PII → Sentiment → Burnout..."):
                result = call_analyze(records, run_topics=run_topics)

            if result and result.get("analyzed"):
                analyzed = result["analyzed"]
                df = pd.DataFrame(analyzed)
                st.session_state["analyzed"] = analyzed
                st.session_state["df"] = df
                st.success(f"Analyzed {result['total_processed']} responses")
                if result.get("validation_errors"):
                    with st.expander(f"Validation warnings ({len(result['validation_errors'])})"):
                        for err in result["validation_errors"]:
                            st.warning(err)

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.divider()
        st.subheader("Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed", len(df))
        c2.metric("Urgent Actions", int((df["action_signal"] == "urgent_action").sum()))
        c3.metric("High/Critical Burnout", int(df["burnout_risk_level"].isin(["high", "critical"]).sum()))
        c4.metric("Avg Burnout Score", f"{df['burnout_risk_score'].mean():.2f}")

        t1, t2, t3 = st.tabs(["Sentiment", "Burnout Risk", "Raw Data"])

        with t1:
            sent_counts = df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["sentiment", "count"]
            fig = px.bar(
                sent_counts, x="sentiment", y="count", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS, title="Sentiment Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            burnout_counts = df["burnout_risk_level"].value_counts().reset_index()
            burnout_counts.columns = ["level", "count"]
            fig = px.pie(
                burnout_counts, names="level", values="count",
                color="level", color_discrete_map=BURNOUT_COLORS,
                title="Burnout Risk Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            urgent_df = df[df["action_signal"] == "urgent_action"][
                ["response_id", "anonymized_text", "sentiment", "burnout_risk_score"]
            ]
            if not urgent_df.empty:
                st.warning(f"{len(urgent_df)} response(s) require urgent attention")
                st.dataframe(urgent_df, use_container_width=True)

        with t3:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Export CSV", csv, "analyzed_responses.csv", "text/csv")


# ── Page: Insights Report ──────────────────────────────────────────────────────

elif page == "Insights Report":
    st.markdown('<p class="main-header">Insights Report</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">LLM-generated executive report with RAG-grounded recommendations</p>',
        unsafe_allow_html=True,
    )

    if "analyzed" not in st.session_state:
        st.info("Run analysis first in the 'Analyze Surveys' page.")
    else:
        if st.button("Generate Insights", type="primary"):
            with st.spinner("Generating with Claude + RAG knowledge base..."):
                report = call_insights(st.session_state["analyzed"])
            if report:
                st.session_state["report"] = report

        if "report" in st.session_state:
            report = st.session_state["report"]

            st.subheader("Executive Summary")
            st.info(report.get("executive_summary", ""))

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Key Findings")
                for finding in report.get("key_findings", []):
                    st.markdown(f"- {finding}")
            with col2:
                bsum = report.get("burnout_risk_summary", {})
                if bsum:
                    fig = px.bar(
                        x=list(bsum.keys()), y=list(bsum.values()),
                        color=list(bsum.keys()), color_discrete_map=BURNOUT_COLORS,
                        labels={"x": "Risk Level", "y": "Count"},
                        title="Burnout Risk Summary",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recommended Actions")
            actions = report.get("recommended_actions", [])
            if actions:
                st.dataframe(pd.DataFrame(actions), use_container_width=True)

            if report.get("rag_references"):
                with st.expander("Knowledge Base References"):
                    for ref in report["rag_references"]:
                        st.markdown(f"> {ref}")

            st.download_button(
                "Export Report JSON",
                json.dumps(report, indent=2).encode(),
                "insights_report.json",
                "application/json",
            )


# ── Page: About ────────────────────────────────────────────────────────────────

elif page == "About":
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    st.markdown("""
**Employee Engagement Intelligence** is a production-ready AI engineering platform for analyzing
employee survey data at scale.

### Pipeline
```
Survey Data -> Data Validation -> PII Anonymization -> Sentiment Analysis
-> Topic Extraction -> Burnout Detection -> RAG Knowledge Base
-> LLM Insights -> FastAPI -> Streamlit -> Docker -> Azure
```

### Tech Stack
| Layer | Technology |
|-------|-----------|
| LLM | Claude (Anthropic) |
| PII Detection | Microsoft Presidio |
| Topic Modeling | BERTopic + multilingual embeddings |
| RAG | ChromaDB |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Cloud | Azure Container Apps |
| Monitoring | Prometheus + Grafana |
""")
