"""
Mini Dolos — Plagiarism Detection API
=======================================
Flask REST API providing all backend services for the Mini Dolos academic
plagiarism detection system.

State Management
-----------------
Analysis state is stored in a module-level dictionary during the lifetime of
the server process. This is intentional for an academic single-session tool.
For a multi-user production deployment, replace with Redis or a database.

API Endpoints
--------------
  POST /upload        — Upload one or more .txt documents
  POST /analyze       — Run the full NLP pipeline (accepts ?threshold=)
  GET  /results       — Return similarity matrix + suspicious pairs
  GET  /heatmap       — Return heatmap PNG as base64
  GET  /network       — Return network graph PNG as base64
  GET  /network-clustered — Return clustered network graph PNG as base64
  GET  /sentences     — Return sentence-level matching results
  GET  /processed     — Return preprocessed token preview per document
  GET  /explanations  — Return shared n-gram explanations per suspicious pair
  GET  /vectors       — Return top-N TF-IDF feature matrix (terms × documents)
  GET  /top-terms     — Return top-10 TF-IDF terms per document
  GET  /vector-space  — Return PCA 2D projection of TF-IDF vectors (base64 PNG)
  GET  /health        — Health check endpoint
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from preprocessing import preprocess_all
from similarity import compute_similarity, get_shared_ngrams
from sentence_matching import match_sentences
from visualization import generate_heatmap, generate_network, generate_pca_plot, generate_clustered_network
from utils import get_file_metadata, get_text_snippet

# ─────────────────────────────────────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB limit

# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Session State
# ─────────────────────────────────────────────────────────────────────────────

state = {
    "documents":    {},     # {filename: raw_text}
    "processed":    {},     # {filename: cleaned_text}
    "metadata":     {},     # {filename: {size_kb, word_count, …}}
    "results":      None,   # output of compute_similarity()
    "explanations": {},     # {docA|||docB: [shared_phrases]}
    "sentences":    {},     # {docA|||docB: [{sentence1, sentence2, score}]}
    "heatmap_b64":  None,   # base64 PNG string
    "network_b64":  None,   # base64 PNG string
    "threshold":    0.75,   # current analysis threshold
}


def _allowed(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _reset_state():
    """Clear all analysis results (called on new upload)."""
    state["documents"].clear()
    state["processed"].clear()
    state["metadata"].clear()
    state["results"] = None
    state["explanations"] = {}
    state["sentences"] = {}
    state["heatmap_b64"] = None
    state["network_b64"] = None


# ─────────────────────────────────────────────────────────────────────────────
# POST /upload
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Accept multiple TXT files via multipart/form-data.

    Validates file extensions, decodes content, stores documents in state,
    and extracts per-file metadata (size, word count).
    Resets all previous analysis state so a fresh session begins.

    Form field: 'files' (multiple)

    Returns 200: {success, files: [{filename, size_kb, word_count, char_count}]}
    Returns 400: {error, details}
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided. Use field name 'files'."}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected."}), 400

    _reset_state()

    uploaded = []
    errors = []

    for f in files:
        if f.filename == "":
            continue
        if not _allowed(f.filename):
            errors.append(f"{f.filename}: Only .txt files are supported.")
            continue

        filename = secure_filename(f.filename)
        try:
            content = f.read().decode("utf-8", errors="replace")
        except Exception as e:
            errors.append(f"{f.filename}: Failed to read — {str(e)}")
            continue

        state["documents"][filename] = content
        meta = get_file_metadata(filename, content)
        state["metadata"][filename] = meta
        uploaded.append(meta)

    if not uploaded:
        return jsonify({"error": "No valid .txt files found.", "details": errors}), 400

    response = {"success": True, "files": uploaded, "errors": errors}
    if len(state["documents"]) < 2:
        response["warning"] = (
            "Upload at least 2 documents to run plagiarism detection."
        )
    return jsonify(response)


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Run the complete NLP plagiarism detection pipeline.

    Pipeline steps:
      1. Preprocess all uploaded documents
      2. Build TF-IDF vectors and compute cosine similarity matrix
      3. Detect suspicious pairs above the threshold
      4. Extract shared n-gram explanations for each suspicious pair
      5. Run sentence-level matching for suspicious pairs
      6. Generate heatmap and network graph visualizations

    Query Parameters:
      threshold (float): Similarity threshold in [0.0, 1.0]. Default 0.75.
                         Can also be passed as JSON body {'threshold': 0.7}.

    Returns 200: {success, message, document_count, suspicious_count, threshold}
    Returns 400: {error}
    Returns 500: {error, details}
    """
    if not state["documents"]:
        return jsonify({"error": "No documents uploaded. Use POST /upload first."}), 400
    if len(state["documents"]) < 2:
        return jsonify({"error": "At least 2 documents required for comparison."}), 400

    # Resolve threshold — query param takes priority over JSON body
    try:
        raw = request.args.get("threshold")
        if raw is None and request.is_json:
            raw = request.json.get("threshold")
        threshold = float(raw) if raw is not None else 0.75
        threshold = max(0.0, min(1.0, threshold))
    except (ValueError, TypeError):
        threshold = 0.75

    state["threshold"] = threshold

    try:
        # Step 1: Preprocessing
        state["processed"] = preprocess_all(state["documents"])

        # Step 2: TF-IDF + Cosine Similarity
        results = compute_similarity(state["processed"], threshold=threshold)
        state["results"] = results

        # Step 3: Shared n-gram explanations
        state["explanations"] = get_shared_ngrams(
            results["vectorizer"],
            results["tfidf_matrix"],
            results["documents"],
            threshold=threshold,
            top_n=10,
        )

        # Step 4: Sentence-level matching
        state["sentences"] = match_sentences(
            state["documents"],
            results["suspicious_pairs"],
            min_words=5,
            sim_threshold=0.6,
        )

        # Step 5: Visualizations
        state["heatmap_b64"] = generate_heatmap(
            results["documents"], results["matrix"]
        )
        state["network_b64"] = generate_network(
            results["documents"], results["suspicious_pairs"]
        )

    except Exception as e:
        return jsonify({"error": "Analysis failed.", "details": str(e)}), 500

    return jsonify({
        "success": True,
        "message": "Analysis complete.",
        "document_count": len(results["documents"]),
        "suspicious_count": len(results["suspicious_pairs"]),
        "threshold": threshold,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /results
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/results", methods=["GET"])
def get_results():
    """
    Return the similarity matrix, suspicious pairs, threshold, and file metadata.

    Returns 200: {documents, matrix, suspicious_pairs, threshold, metadata}
    Returns 404: {error}
    """
    if state["results"] is None:
        return jsonify({"error": "No results. Run POST /analyze first."}), 404

    return jsonify({
        "documents":       state["results"]["documents"],
        "matrix":          state["results"]["matrix"],
        "suspicious_pairs": state["results"]["suspicious_pairs"],
        "threshold":       state["threshold"],
        "metadata":        state["metadata"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /heatmap
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/heatmap", methods=["GET"])
def get_heatmap():
    """
    Return the similarity heatmap as a base64-encoded PNG string.

    Returns 200: {image: '<base64 string>'}
    Returns 404: {error}
    """
    if state["heatmap_b64"] is None:
        return jsonify({"error": "No heatmap. Run POST /analyze first."}), 404
    return jsonify({"image": state["heatmap_b64"]})


# ─────────────────────────────────────────────────────────────────────────────
# GET /network
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/network", methods=["GET"])
def get_network():
    """
    Return the similarity network graph as a base64-encoded PNG string.

    Returns 200: {image: '<base64 string>'}
    Returns 404: {error}
    """
    if state["network_b64"] is None:
        return jsonify({"error": "No network graph. Run POST /analyze first."}), 404
    return jsonify({"image": state["network_b64"]})


# ─────────────────────────────────────────────────────────────────────────────
# GET /sentences
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/sentences", methods=["GET"])
def get_sentences():
    """
    Return sentence-level matching results for suspicious document pairs.

    Returns 200: {matches: {'docA|||docB': [{sentence1, sentence2, score}]}}
    Returns 404: {error}
    """
    if not state["sentences"]:
        return jsonify({"error": "No sentence analysis. Run POST /analyze first."}), 404
    return jsonify({"matches": state["sentences"]})


# ─────────────────────────────────────────────────────────────────────────────
# GET /processed
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/processed", methods=["GET"])
def get_processed():
    """
    Return preprocessed token previews for each uploaded document.

    Allows evaluators to verify the preprocessing pipeline output.
    Returns the first 100 tokens of each document alongside the
    original text snippet for comparison.

    Returns 200: {filename: {tokens, token_count, original_snippet}}
    Returns 404: {error}
    """
    if not state["processed"]:
        return jsonify({"error": "No processed data. Run POST /analyze first."}), 404

    preview = {}
    for filename, text in state["processed"].items():
        words = text.split()
        preview[filename] = {
            "tokens": " ".join(words[:100]) + ("..." if len(words) > 100 else ""),
            "token_count": len(words),
            "original_snippet": get_text_snippet(
                state["documents"].get(filename, ""), 250
            ),
        }
    return jsonify(preview)


# ─────────────────────────────────────────────────────────────────────────────
# GET /explanations
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/explanations", methods=["GET"])
def get_explanations():
    """
    Return shared n-gram explanations for each suspicious document pair.

    Provides interpretable evidence: the specific vocabulary phrases that
    caused two documents to be flagged as similar.

    Returns 200: {'docA|||docB': ['phrase_1', 'phrase_2', ...], ...}
    Returns 404: {error}
    """
    if not state["explanations"]:
        return jsonify({"error": "No explanations. Run POST /analyze first."}), 404
    return jsonify(state["explanations"])


# ─────────────────────────────────────────────────────────────────────────────
# GET /vectors
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/vectors", methods=["GET"])
def get_vectors():
    """
    Return the TF-IDF feature matrix limited to the top-N most weighted terms.

    Documents are represented as high-dimensional sparse vectors in TF-IDF space.
    Returning the full vocabulary (often thousands of features) would produce an
    unreadably wide table, so we select the top-N features by their maximum
    TF-IDF weight across any document — these are the most discriminative terms.

    Query Parameters:
      top_n (int): Number of terms to return. Default 40. Max 100.

    Returns 200:
      {
        "terms":          ["machine learning", "cosine similarity", ...],
        "vectors":        {"doc1.txt": [0.32, 0.44, ...], ...},
        "total_features": 1243,
        "shown_features": 40
      }
    Returns 404: {error}
    """
    if state["results"] is None:
        return jsonify({"error": "No results. Run POST /analyze first."}), 404

    vectorizer   = state["results"]["vectorizer"]
    tfidf_matrix = state["results"]["tfidf_matrix"]
    documents    = state["results"]["documents"]

    try:
        top_n = int(request.args.get("top_n", 40))
        top_n = max(5, min(100, top_n))
    except (ValueError, TypeError):
        top_n = 40

    feature_names = vectorizer.get_feature_names_out()
    dense         = tfidf_matrix.toarray()

    # Select top_n terms by maximum weight across all documents
    # These are the terms most responsible for differentiating documents.
    max_weights = dense.max(axis=0)          # shape: (n_features,)
    top_indices = np.argsort(max_weights)[::-1][:top_n]

    top_terms = [feature_names[i] for i in top_indices]
    vectors   = {
        doc: [round(float(dense[j][i]), 4) for i in top_indices]
        for j, doc in enumerate(documents)
    }

    return jsonify({
        "terms":          top_terms,
        "vectors":        vectors,
        "total_features": int(len(feature_names)),
        "shown_features": top_n,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /top-terms
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/top-terms", methods=["GET"])
def get_top_terms():
    """
    Return the top-10 TF-IDF weighted terms per document.

    High-weight terms are the vocabulary items most characteristic of each
    document. Comparing these lists across documents explains why two documents
    score high on cosine similarity — they share the same high-weight vocabulary.

    Query Parameters:
      top_n (int): Terms per document. Default 10. Max 20.

    Returns 200:
      {
        "doc1.txt": [["machine learning", 0.44], ["neural network", 0.38], ...],
        ...
      }
    Returns 404: {error}
    """
    if state["results"] is None:
        return jsonify({"error": "No results. Run POST /analyze first."}), 404

    vectorizer   = state["results"]["vectorizer"]
    tfidf_matrix = state["results"]["tfidf_matrix"]
    documents    = state["results"]["documents"]

    try:
        top_n = int(request.args.get("top_n", 10))
        top_n = max(5, min(20, top_n))
    except (ValueError, TypeError):
        top_n = 10

    feature_names = vectorizer.get_feature_names_out()
    dense         = tfidf_matrix.toarray()

    top_terms = {}
    for j, doc in enumerate(documents):
        weights     = dense[j]
        top_indices = np.argsort(weights)[::-1][:top_n]
        top_terms[doc] = [
            [feature_names[i], round(float(weights[i]), 4)]
            for i in top_indices
            if weights[i] > 0.0       # Skip zero-weight features
        ]

    return jsonify(top_terms)


# ─────────────────────────────────────────────────────────────────────────────
# GET /vector-space
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/vector-space", methods=["GET"])
def get_vector_space():
    """
    Return a PCA 2D projection of TF-IDF document vectors as a base64 PNG.

    TF-IDF vectors live in a high-dimensional space (one axis per vocabulary
    term). PCA reduces this to 2 principal components that capture the most
    variance, allowing documents to be plotted in a 2D scatter plot.

    Documents that are close together in this plot share similar TF-IDF
    distributions — i.e., they discuss similar topics in similar proportions.
    This directly explains why they receive high cosine similarity scores.

    Returns 200: {image: '<base64 PNG string>'}
    Returns 404: {error}
    Returns 500: {error, details}
    """
    if state["results"] is None:
        return jsonify({"error": "No results. Run POST /analyze first."}), 404

    try:
        image = generate_pca_plot(
            state["results"]["documents"],
            state["results"]["tfidf_matrix"],
            state["results"]["suspicious_pairs"],
        )
    except Exception as e:
        return jsonify({"error": "PCA plot generation failed.", "details": str(e)}), 500

    return jsonify({"image": image})


# ─────────────────────────────────────────────────────────────────────────────
# GET /network-clustered
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/network-clustered", methods=["GET"])
def get_network_clustered():
    """
    Return a color-coded clustered document similarity network graph as a base64 PNG.

    Query Parameters:
      threshold (float): Only edges > threshold are shown. Default 0.75.
      hide_singletons (bool): If true, remove isolated nodes. Default false.

    Returns 200: {image: '<base64 PNG string>'}
    Returns 404: {error}
    Returns 500: {error, details}
    """
    if state["results"] is None:
        return jsonify({"error": "No results. Run POST /analyze first."}), 404

    try:
        threshold = float(request.args.get("threshold", 0.75))
    except ValueError:
        threshold = 0.75

    hide_singletons_str = request.args.get("hide_singletons", "false").lower()
    hide_singletons = hide_singletons_str == "true"

    try:
        # Use the sparse tfidf_matrix internally through score checking instead of passing matrix
        # Let's pass the matrix from the state results.
        matrix = state["results"]["matrix"]
        image = generate_clustered_network(
            state["results"]["documents"],
            matrix,
            threshold=threshold,
            hide_singletons=hide_singletons,
        )
    except Exception as e:
        return jsonify({"error": "Clustered Network graph generation failed.", "details": str(e)}), 500

    return jsonify({"image": image})


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — confirms the API is running."""
    return jsonify({
        "status": "ok",
        "service": "Mini Dolos Plagiarism Detection API",
        "documents_loaded": len(state["documents"]),
        "analysis_ready": state["results"] is not None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Mini Dolos — Plagiarism Detection API")
    print("  Running on http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
