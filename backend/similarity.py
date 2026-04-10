"""
TF-IDF Vectorization and Cosine Similarity
===========================================
This module converts preprocessed document text into TF-IDF feature vectors
and computes pairwise cosine similarity — the mathematical core of Mini Dolos.

═══════════════════════════════════════════════════════════════════════════════
PART 1 — How TF-IDF Represents Documents as Numeric Vectors
═══════════════════════════════════════════════════════════════════════════════

Each document is converted into a numeric vector in a high-dimensional space.
Each dimension of this space corresponds to one vocabulary term (or n-gram
phrase) from the entire corpus.

For a term t in document d within corpus D (N total documents):

    TF(t, d)     = count(t in d) / total_terms(d)
                   Proportion of the document occupied by this term.

    IDF(t)       = log( (N + 1) / (df(t) + 1) ) + 1     [smooth variant]
                   Inverse frequency across documents. Terms used in every
                   document get a low IDF (they are uninformative). Terms
                   used in only one document get a high IDF (distinctive).

    TF-IDF(t, d) = TF(t, d) × IDF(t)
                   Combined weight. High weight = this term is both frequent
                   in this document AND rare across the corpus.

After computing raw TF-IDF weights, sklearn L2-normalises each document
vector so that ||v|| = 1. This makes the vectors direction-only, removing
the effect of document length.

Final vector space:
  - Each document d → a vector vd ∈ ℝ^V   (V = vocabulary size)
  - vd[i] = TF-IDF weight of feature i in document d (after L2 norm)
  - Most entries are 0 (sparse: most terms appear in only some documents)

Why TF-IDF outperforms raw counts for plagiarism detection:
  Raw counts give equal importance to common words ("the", "is", "and") and
  domain-specific terms ("neural network", "gradient descent"). TF-IDF
  suppresses common vocabulary via the IDF factor and amplifies distinctive
  vocabulary. A plagiarist who copies the phrase "cosine similarity score" will
  have that phrase flagged because it is rare across the corpus but prominent
  in both the original and copied document.

sublinear_tf=True in this implementation:
  Replaces TF with log(1 + TF) to dampen the effect of extremely frequent
  terms. Without this, one very repeated word would dominate the vector.

═══════════════════════════════════════════════════════════════════════════════
PART 2 — Why n-grams (1, 2) Improve Vector Representation Quality
═══════════════════════════════════════════════════════════════════════════════

ngram_range=(1, 2) means features include both unigrams and bigrams:

  Unigrams (n=1):   each individual word is a separate dimension.
    Examples: "machine", "learning", "network", "similarity"

  Bigrams (n=2):    each consecutive word-pair is a separate dimension.
    Examples: "machine learning", "neural network", "cosine similarity"

Why bigrams matter for plagiarism:
  Consider these two sentences:
    Original:   "machine learning algorithms classify patterns"
    Paraphrase: "learning algorithms classify machine patterns"
  - Unigrams: IDENTICAL (same words, different order) → false positive
  - Bigrams:  DIFFERENT ("machine learning" vs "learning algorithms")
              Bigrams capture phrase structure and word order.

  Conversely, when phrases ARE copied verbatim:
    Original:   "gradient descent optimizes the loss function"
    Plagiarised: "gradient descent was used to optimize the loss function"
  - Bigrams "gradient descent" and "loss function" are shared.
    These become high-weight shared features, correctly raising similarity.

Vocabulary explosion is manageable:
  With (1,2) n-grams, vocabulary grows significantly but is still sparse.
  min_df=1 ensures every feature is included (appropriate for small academic
  corpora of 2-20 assignments).

═══════════════════════════════════════════════════════════════════════════════
PART 3 — How Cosine Similarity Compares Document Vectors
═══════════════════════════════════════════════════════════════════════════════

After L2 normalisation, ||vA|| = ||vB|| = 1, so:

    cos(θ) = (vA · vB) / (||vA|| × ||vB||)
           = vA · vB                              (since norms are both 1)
           = Σᵢ (vA[i] × vB[i])

Geometric interpretation:
  - cos(θ) measures the ANGLE between two document vectors, not their
    Euclidean distance. Two vectors pointing in the same direction → θ=0°
    → cos(0°)=1.0 (identical content distribution).
  - Two documents discussing completely different topics → nearly orthogonal
    vectors → θ ≈ 90° → cos(90°) ≈ 0.0.

Why we use cosine and not Euclidean distance:
  Euclidean distance ||vA - vB|| is affected by document length. A long
  document has many non-zero weights, making it "far" from a short document
  even if they discuss the same topic. Cosine similarity is length-invariant
  because we measure angle, not magnitude difference. This makes it the
  standard metric for document similarity in information retrieval.

Threshold interpretation:
  cos(θ) > 0.75  → documents share ~75% of weighted vocabulary direction
                   → suspicious: likely involves copied or closely paraphrased content
  cos(θ) > 0.90  → very high : almost certainly copied with minor edits
  cos(θ) > 0.50  → moderate  : topic overlap, some shared vocabulary
  cos(θ) < 0.30  → low       : different topics, negligible overlap

PCA projection (see visualization.py):
  Since vectors live in ℝ^V (thousands of dimensions), we use PCA to project
  to ℝ² for visualisation. Direction in ℝ² approximates direction in ℝ^V —
  documents close together in the 2D scatter plot will have high cosine
  similarity, and documents far apart will have low cosine similarity.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def compute_similarity(processed_docs: dict, threshold: float = 0.75) -> dict:
    """
    Build TF-IDF feature vectors and compute the pairwise cosine similarity matrix.

    Args:
        processed_docs: {filename: cleaned_preprocessed_text}
                        Output from preprocessing.preprocess_all().
        threshold:      Minimum cosine similarity score to flag a document pair
                        as suspicious. Default 0.75 (75% similarity).

    Returns:
        {
          'documents':        [ordered list of filenames],
          'matrix':           [[cosine similarity scores, 4 d.p.]],  ← n×n symmetric
          'suspicious_pairs': [{'doc1', 'doc2', 'score'}, ...],      ← sorted desc
          'vectorizer':       fitted TfidfVectorizer instance,
          'tfidf_matrix':     sparse scipy matrix of TF-IDF features
        }
    """
    filenames = list(processed_docs.keys())
    texts = [processed_docs[f] for f in filenames]

    # Build TF-IDF feature matrix
    # ngram_range=(1,2):  include unigrams and bigrams as features
    # min_df=1:           include terms appearing in ≥1 document (small corpus)
    # smooth_idf=True:    add 1 to df to prevent log(0) for unseen terms
    # sublinear_tf=True:  apply log(1 + tf) to dampen effect of repeated terms
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        smooth_idf=True,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute pairwise cosine similarity: cos(θ) = (A · B) / (|A| × |B|)
    # sklearn handles L2 normalization automatically before computing dot products
    # Result: n×n symmetric matrix with values in [0, 1]
    sim_matrix = sklearn_cosine(tfidf_matrix)

    # Collect suspicious pairs from the upper triangle (avoids duplicates)
    suspicious = []
    n = len(filenames)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i][j])
            if score > threshold:
                suspicious.append({
                    "doc1": filenames[i],
                    "doc2": filenames[j],
                    "score": round(score, 4),
                })

    # Sort by similarity descending (highest-risk pairs shown first)
    suspicious.sort(key=lambda x: x["score"], reverse=True)

    return {
        "documents": filenames,
        "matrix": [[round(float(v), 4) for v in row] for row in sim_matrix],
        "suspicious_pairs": suspicious,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }


def get_shared_ngrams(
    vectorizer,
    tfidf_matrix,
    filenames: list,
    threshold: float = 0.75,
    top_n: int = 10,
) -> dict:
    """
    Extract the top shared n-gram phrases responsible for similarity between
    suspicious document pairs — providing interpretable evidence for flagged cases.

    Method (TF-IDF vocabulary intersection):
      For each suspicious pair (doc_i, doc_j):
        1. Find all vocabulary features (n-grams) where both documents have
           non-zero TF-IDF weights (i.e., the term appears in both docs).
        2. Rank shared features by combined TF-IDF weight: high combined weight
           means the term is distinctively important in both documents.
        3. Return the top_n features as the 'explanation' for the similarity.
        4. Bigrams are surfaced first since multi-word phrases provide more
           specific, readable evidence than single-word matches.

    Args:
        vectorizer:    Fitted TfidfVectorizer from compute_similarity().
        tfidf_matrix:  Sparse TF-IDF matrix from compute_similarity().
        filenames:     Ordered document names (must match tfidf_matrix rows).
        threshold:     same threshold used in compute_similarity().
        top_n:         Number of top shared phrases to return per pair.

    Returns:
        {'docA|||docB': ['phrase_1', 'phrase_2', ...], ...}
        Keys use '|||' as a safe delimiter between document names.
    """
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.toarray()
    n = len(filenames)
    sim_matrix = sklearn_cosine(tfidf_matrix)
    explanations = {}

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i][j])
            if score <= threshold:
                continue

            key = f"{filenames[i]}|||{filenames[j]}"

            # Intersection: features with non-zero weight in BOTH documents
            shared_mask = (dense[i] > 0) & (dense[j] > 0)
            shared_indices = np.where(shared_mask)[0]

            if len(shared_indices) == 0:
                explanations[key] = []
                continue

            # Rank by combined TF-IDF weight (higher = more distinctive in both)
            combined = [
                (feature_names[idx],
                 float(dense[i][idx]) + float(dense[j][idx]))
                for idx in shared_indices
            ]
            combined.sort(key=lambda x: x[1], reverse=True)

            # Surface bigrams first for readability, then unigrams
            phrases = [p for p, _ in combined]
            bigrams = [p for p in phrases if " " in p]
            unigrams = [p for p in phrases if " " not in p]
            explanations[key] = (bigrams + unigrams)[:top_n]

    return explanations
