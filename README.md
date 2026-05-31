# 🌐 Multilingual Employee Engagement Survey Analysis with LLMs

> **LLM-powered NLP pipeline** for analyzing employee engagement survey responses across 6 languages (English, German, French, Spanish, Hindi, Chinese) using Claude AI, BERTopic, and multilingual transformer embeddings.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Claude API](https://img.shields.io/badge/LLM-Claude%20Sonnet-orange.svg)](https://www.anthropic.com/)
[![BERTopic](https://img.shields.io/badge/Topics-BERTopic-green.svg)](https://maartengr.github.io/BERTopic/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Project Does

Employee engagement surveys generate thousands of open-text responses — often in **multiple languages** in global organizations. Traditional keyword-based sentiment tools fail here.

This pipeline:
1. **Generates** a realistic synthetic multilingual survey dataset (via Claude API)
2. **Preprocesses** text with language detection and validation
3. **Analyzes** each response with Claude: sentiment, themes, key phrases, action signals
4. **Models** latent topics using BERTopic with multilingual embeddings
5. **Visualizes** everything in an interactive Streamlit dashboard

---

## 🏗️ Architecture

```
Survey Responses (6 languages)
         │
         ▼
┌─────────────────────┐
│  Data Generation    │  ← Claude API generates realistic synthetic data
│  (src/data_gen)     │    across 10 engagement themes × 6 languages
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Preprocessing      │  ← langdetect + lingua, unicode normalization,
│  (src/preprocessing)│    language validation, text features
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM Analysis       │  ← Batch Claude API calls (10 responses/call)
│  (src/llm_analyzer) │    → sentiment (5-class) + confidence
└────────┬────────────┘    → primary/secondary themes
         │                 → key phrases (in original language)
         ▼                 → English summary
┌─────────────────────┐    → emotion tags + action signal
│  Topic Modeling     │
│  (src/topic_modeling│  ← BERTopic + paraphrase-multilingual-MiniLM-L12-v2
└────────┬────────────┘    discovers latent topics across all languages
         │
         ▼
┌─────────────────────┐
│  Streamlit Dashboard│  ← Interactive visualization of all results
│  (app/streamlit_app)│    with language, theme, sentiment filters
└─────────────────────┘
```

---

## 📊 Dataset

### Why Synthetic?

No public dataset exists with multilingual **employee engagement survey** free-text responses across all 6 target languages. Options evaluated:

| Source | Problem |
|--------|---------|
| Kaggle employee surveys | English only, small |
| Glassdoor reviews | Legal risk, not open-text survey format |
| Amazon multilingual reviews | Wrong domain (products, not HR) |
| **Synthetic via Claude** | ✅ Correct domain, all 6 languages, controlled distribution |

### Dataset Schema

| Column | Description |
|--------|-------------|
| `response_text` | Raw survey response (in original language) |
| `language_code` | ISO 639-1 code (en, de, fr, es, hi, zh) |
| `theme_key` | Ground-truth engagement theme |
| `sentiment_label` | Ground-truth sentiment (5-class) |
| `department` | Simulated department |
| `experience_level` | Simulated tenure |
| `llm_sentiment` | Claude-predicted sentiment |
| `llm_primary_theme` | Claude-detected theme |
| `llm_key_phrases` | Key phrases extracted (in original language) |
| `llm_summary_en` | English summary |
| `llm_action_signal` | `urgent_action` / `monitor` / `positive_share` / `no_action` |
| `engagement_score` | Numeric 0–100 from sentiment |
| `topic_id` | BERTopic cluster ID |

### Engagement Themes (10)

| Key | Theme |
|-----|-------|
| `workload_balance` | Workload & Work-Life Balance |
| `management_leadership` | Management & Leadership Quality |
| `recognition_rewards` | Recognition & Rewards |
| `career_growth` | Career Development & Growth |
| `team_collaboration` | Team Collaboration & Culture |
| `company_strategy` | Company Strategy & Direction |
| `tools_resources` | Tools, Resources & Environment |
| `inclusion_belonging` | Diversity, Inclusion & Belonging |
| `wellbeing_safety` | Employee Wellbeing & Psychological Safety |
| `communication` | Internal Communication & Transparency |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/employee-engagement-llm.git
cd employee-engagement-llm
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=your_key_here
```

### 3. Test Run (100 responses, no API cost for topics)

```bash
# Generate 50 synthetic responses per language (300 total) — fast test
python run_pipeline.py generate --samples 50

# Preprocess
python run_pipeline.py preprocess

# Analyze with Claude (first 50 rows only, for testing)
python run_pipeline.py analyze --limit 50

# Launch dashboard (works even with partial data)
streamlit run app/streamlit_app.py
```

### 4. Full Pipeline

```bash
python run_pipeline.py all --samples 500
# Generates 3000 responses total, analyzes all, runs topic modeling
# Estimated API cost: ~$2-4 with Claude Sonnet
```

---

## 💡 Key Technical Decisions

### Claude as the LLM backbone

- **Batching**: 10 responses per API call → reduces cost ~10x vs individual calls
- **Structured output**: System prompt enforces JSON schema with pydantic-style validation
- **Multilingual native**: Claude handles all 6 languages without translation step
- **Action signals**: Custom business logic (`urgent_action`) beyond simple sentiment

### BERTopic + Multilingual Embeddings

- Model: `paraphrase-multilingual-MiniLM-L12-v2` (420MB, no GPU needed)
- Embedding cache: Computed once, reused across runs
- UMAP: `n_components=5`, `metric=cosine` — tuned for short survey text
- HDBSCAN: `min_cluster_size=10` — avoids micro-clusters in HR survey data

### Language Detection Strategy

- **Primary**: `lingua-language-detector` — better for short text, non-Latin scripts
- **Fallback**: `langdetect` — broader language coverage
- **Validation**: Flag language mismatches without dropping (optional filter)

---

## 📁 Project Structure

```
employee-engagement-llm/
├── src/
│   ├── config.py              # All constants (themes, languages, paths, API config)
│   ├── data_generation.py     # Synthetic dataset generation via Claude API
│   ├── preprocessing.py       # Language detection, cleaning, text features
│   ├── llm_analyzer.py        # Batch Claude analysis → structured sentiment/themes
│   └── topic_modeling.py      # BERTopic multilingual topic discovery
├── app/
│   └── streamlit_app.py       # Interactive dashboard
├── data/
│   ├── raw/                   # External datasets (HuggingFace)
│   ├── synthetic/             # Claude-generated survey responses
│   └── processed/             # Cleaned, analyzed, topic-tagged data
├── notebooks/                 # Jupyter EDA notebooks (see below)
├── run_pipeline.py            # Single entry point for all steps
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_dataset_creation.ipynb` | Walkthrough of synthetic data generation |
| `02_eda_language_distribution.ipynb` | Language distribution, text length, quality checks |
| `03_llm_sentiment_analysis.ipynb` | Sentiment results, confidence calibration |
| `04_topic_modeling.ipynb` | BERTopic visualization, topic-theme mapping |
| `05_dashboard_insights.ipynb` | Aggregated HR insights and recommendations |

---

## 🖥️ Dashboard Preview

The Streamlit dashboard offers:
- **KPI cards**: Engagement score, positive rate, urgent signal count
- **Sentiment distribution** with interactive filtering
- **Language × Theme heatmap** (engagement score matrix)
- **Action items tab**: Urgent signals + positive highlights to share
- **Response explorer**: Search, filter, sort individual responses

---

## 🔮 Potential Extensions

- [ ] Fine-tune a smaller model (DistilBERT) on the generated dataset
- [ ] Add RAG: retrieve similar historical responses for context
- [ ] Time-series analysis if survey rounds are dated
- [ ] Multilingual keyword extraction with KeyBERT
- [ ] Export PDF reports per department/language

---

## ⚠️ Ethical Considerations

- All data is **synthetic** — no real employee data used
- Responses are generated to be **anonymous by design**
- LLM analysis is a **decision support tool**, not a replacement for HR judgment
- Urgent action signals require **human review** before any action

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

**Vedant Tele** — Data Analyst & AI Engineering practitioner  
People Analytics | Automation | NLP  
Munich, Germany
