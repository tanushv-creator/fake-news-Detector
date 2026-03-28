# Fake News Detector

A machine learning project that classifies news articles as **real** or **fake** using a TF-IDF + Random Forest pipeline.

---

## Features

- Fetches the WELFake benchmark dataset (~72k articles) directly from the web
- Text preprocessing: lowercasing, URL removal, stopword filtering, stemming
- TF-IDF vectorization with unigrams + bigrams
- Random Forest classifier (200 trees, balanced class weights)
- Evaluation: accuracy, ROC-AUC, confusion matrix, classification report
- Feature importance visualization (top TF-IDF terms)
- Live prediction — input raw text **or** a news article URL

---

## Project Structure

```
fake-news-detector/
├── fake_news_detector.ipynb   # Main notebook (run top to bottom)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook fake_news_detector.ipynb
```

Run all cells from top to bottom. The notebook will:
- Download the dataset automatically
- Train the model
- Print evaluation metrics
- Save plots and the trained model

---

## Usage — Live Predictions

After training, use the `predict_news()` function in the last section:

```python
# From text
predict_news(text="Oxford researchers publish new vaccine trial results in The Lancet...")

# From URL
predict_news(url="https://www.bbc.com/news/science-environment-68886769")
```

**Output:**
```
=============================================
 ✅  VERDICT: REAL
    Credibility Score : 82.3/100
    Real probability  : 82.3%
    Fake probability  : 17.7%
    Confidence        : High
=============================================
```

---

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~95%   |
| ROC-AUC   | ~0.98  |

> Results may vary depending on dataset version and random seed.

---

## Dataset

**WELFake** — A dataset of 72,134 news articles (35,028 real + 37,106 fake).  
Source: Verma, P.K. et al., *WELFake: Word Embedding over Linguistic Features for Fake News Detection* (2021).

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `scikit-learn` | TF-IDF + Random Forest |
| `nltk` | Text preprocessing |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualizations |
| `joblib` | Model serialization |

---

## License

MIT License — free to use and modify for academic and personal projects.
