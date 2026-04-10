# Mini Dolos — Plagiarism Detection System using Cosine Similarity

An academic full-stack plagiarism detection tool that identifies similarity between student assignment documents using **TF-IDF vectorization** and **cosine similarity**.

Inspired by [Dolos](https://dolos.ugent.be/), built for educational NLP coursework.

---

## Features

| Feature | Description |
|---|---|
| Multi-file Upload | Upload any number of `.txt` student assignments |
| TF-IDF Vectorization | `sklearn` TfidfVectorizer with bigram support |
| Cosine Similarity | Pairwise similarity matrix between all documents |
| Configurable Threshold | Adjustable detection threshold (60%–90%, default 75%) |
| Similarity Heatmap | `seaborn` annotated similarity matrix chart |
| Network Graph | `networkx` document similarity graph |
| Sentence Matching | Sentence-level matching with minimum word filter |
| Shared N-gram Explanations | Top shared phrases per suspicious pair |
| Preprocessing Preview | View cleaned tokens vs. original text |
| CSV Export | Download similarity matrix as CSV |

---

## Expected Similarity (Sample Dataset)

| Pair | Expected |
|---|---|
| assignment1 vs assignment2 | **~82%** — Partially copied |
| assignment1 vs assignment3 | **~45%** — Paraphrased |
| assignment1 vs assignment4 | **~12%** — Unrelated topic |
| assignment1 vs assignment5 | **~90%** — Heavily copied |

---

## Project Structure

```
Nlp/
├── backend/
│   ├── app.py               # Flask API (8 endpoints)
│   ├── preprocessing.py     # NLP pipeline: lowercase → remove stopwords → lemmatize
│   ├── similarity.py        # TF-IDF vectorization + cosine similarity
│   ├── sentence_matching.py # Sentence-level similarity with noise filtering
│   ├── visualization.py     # Seaborn heatmap + networkx graph
│   ├── utils.py             # Helper functions
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── UploadPage.jsx        # File upload + analysis trigger
│       │   ├── ResultsPage.jsx       # 6-tab results dashboard
│       │   └── VisualizationPage.jsx # Heatmap + network graph
│       ├── components/
│       │   ├── Sidebar.jsx           # Left navigation
│       │   └── PipelineStatus.jsx    # Step-by-step progress indicator
│       ├── context/
│       │   └── AnalysisContext.jsx   # Global React state (useReducer)
│       └── services/
│           └── api.js                # Axios API wrapper
├── sample_data/
│   ├── assignment1.txt   # Original essay
│   ├── assignment2.txt   # Partially copied (~82% similar to 1)
│   ├── assignment3.txt   # Paraphrased    (~45% similar to 1)
│   ├── assignment4.txt   # Unrelated      (~12% similar to 1)
│   └── assignment5.txt   # Heavily copied (~90% similar to 1)
└── README.md
```

---

## Setup Instructions

### Backend

**Requires Python 3.8+**

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python app.py
```

The API will run at **http://localhost:5000**

### Frontend

**Requires Node.js 18+**

```bash
cd frontend
npm install
npm run dev
```

The app will run at **http://localhost:5173**

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload `.txt` files (field: `files`) |
| POST | `/analyze?threshold=0.75` | Run full NLP pipeline |
| GET | `/results` | Similarity matrix + suspicious pairs |
| GET | `/heatmap` | Heatmap PNG as base64 |
| GET | `/network` | Network graph PNG as base64 |
| GET | `/sentences` | Sentence-level match results |
| GET | `/processed` | Preprocessed token previews |
| GET | `/explanations` | Shared n-gram explanations |
| GET | `/health` | Health check |

---

## NLP Pipeline Explanation

### 1. Preprocessing (`preprocessing.py`)
```
raw text
  → lowercase           (normalize capitalization)
  → remove punctuation  (eliminate noise characters)
  → tokenize            (split into words using NLTK Punkt)
  → remove stopwords    (remove "the", "is", "and", ...)
  → lemmatize           ("running" → "run", "algorithms" → "algorithm")
  → cleaned string
```

### 2. TF-IDF Vectorization (`similarity.py`)
```
TF(t,d)       = count(t in d) / total_terms(d)
IDF(t)        = log(N / df(t))
TF-IDF(t,d)   = TF(t,d) × IDF(t)

ngram_range=(1,2)  → captures individual words AND phrases
```

### 3. Cosine Similarity
```
cos(θ) = (A · B) / (|A| × |B|)

Range: [0.0, 1.0]
1.0 = identical content
0.0 = completely different content
```

### 4. Suspicious Pair Detection
```
if cosine_similarity(doc_i, doc_j) > threshold:
    flag as potential plagiarism
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · Flask · flask-cors |
| NLP | NLTK · scikit-learn · NumPy · Pandas |
| Visualization | seaborn · matplotlib · networkx |
| Frontend | React 18 · Vite · TailwindCSS v3 |
| HTTP | Axios |

---

## Academic Purpose

This system is designed for an **NLP course project** to demonstrate:
- Text preprocessing pipelines
- TF-IDF feature extraction
- Cosine similarity for document comparison
- Information retrieval concepts applied to plagiarism detection
