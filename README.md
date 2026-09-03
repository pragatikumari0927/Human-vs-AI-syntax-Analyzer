# 🔤 Human vs AI Syntax Analyzer

A Streamlit app for comparing the syntactic structure of human-written vs AI-generated text, built on spaCy dependency parsing.

## Features

- **📊 Corpus Dashboard** — upload CSVs exported from `Human_vs_AI_Syntax_Corpus.ipynb` and explore descriptive statistics, significance/effect-size tables (Mann-Whitney U + Cohen's d), and interactive charts.
- **✍️ Live Analysis** — paste any text and get instant sentence-level syntactic features (POS ratios, dependency tree depth, clause count, passive voice, dependency distance), computed live with the same spaCy pipeline used to build the corpus. If a corpus is loaded in the Dashboard tab, your text is compared against the human/AI corpus averages.

## Project structure

```
.
├── app.py              # Streamlit app (UI + tabs)
├── utils.py            # Shared syntactic feature extraction logic
├── requirements.txt    # Python dependencies
└── README.md
```

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` already includes the `en_core_web_sm` spaCy model as a direct wheel download, so a separate `python -m spacy download` step isn't needed. If you'd rather install it the standard way, remove the last line from `requirements.txt` and run:
> ```bash
> python -m spacy download en_core_web_sm
> ```

### 3. Run the app

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

### 4. Get data for the Dashboard tab

Run `Human_vs_AI_Syntax_Corpus.ipynb` (in Google Colab or locally) through the export section, then download:

- `human_ai_syntactic_features.csv` — required
- `human_ai_statistical_comparison.csv` — optional (speeds up loading; otherwise the app recomputes the same statistics itself)

Upload both files in the **Corpus Dashboard** tab.

## Deploying

To share the app without running it locally:

1. Push this repo to GitHub (see below if you're new to Git).
2. Go to [share.streamlit.io](https://share.streamlit.io) (Streamlit Community Cloud), sign in with GitHub, and deploy this repo. Point it at `app.py`.

`requirements.txt` is already set up so the spaCy model installs correctly on the server.

## Pushing this project to GitHub (first time)

If this is your first time using Git/GitHub:

1. Create a new empty repository on [github.com](https://github.com/new) — **don't** initialize it with a README (you already have one).
2. In this project folder, run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
3. Refresh your GitHub repo page — your files should appear.
