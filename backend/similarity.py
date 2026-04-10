"""
TF-IDF Vectorization and Cosine Similarity
===========================================
This module converts preprocessed document text into TF-IDF feature vectors
and computes pairwise cosine similarity—the mathematical core of Mini Dolos.

TF-IDF Explanation (Term Frequency – Inverse Document Frequency)
-----------------------------------------------------------------
For a term t in document d within corpus D of N documents:

    TF(t, d)  = count(t in d) / total_terms(d)
    IDF(t)    = log( N / df(t) )     where df(t) = documents containing t
    TF-IDF(t, d) = TF(t, d) × IDF(t)

Effect:
  - Terms frequent in one doc but rare across others → high weight
  - Terms appearing in every document (like "the") → IDF ≈ 0 → near-zero weight

Why TF-IDF outperforms raw term counts for plagiarism detection:
  Raw counts give equal importance to high-frequency generic words and rare
  distinctive terms. TF-IDF suppresses ubiquitous words and amplifies the
  weight of distinctive, domain-specific vocabulary, making it far more
  sensitive to the kind of copied or paraphrased content that constitutes
  academic plagiarism.

Why n-grams (1, 2) improve phrase detection:
  - Unigrams (n=1) — capture individual word matches
      e.g., "machine", "learning", "neural", "dataset"
  - Bigrams (n=2)  — capture sequential phrase matches
      e.g., "machine_learning", "neural_network", "training_dataset"
  Together, (1,2) n-grams detect both word-level and phrase-level copying,
  essential for identifying partial paraphrasing where sentences are
  restructured but key phrases are preserved verbatim.

Cosine Similarity
-----------------
Measures the cosine of the angle θ between two TF-IDF vectors A and B:

    cos(θ) = (A · B) / (|A| × |B|)

  Where:
    A · B  = Σ(Aᵢ × Bᵢ)   — dot product (sum of element-wise products)
    |A|    = √Σ(Aᵢ²)       — L2 norm (Euclidean length) of vector A

Range: [0.0, 1.0]
  1.0 → vectors point in identical direction → identical vocabulary distribution
  0.0 → vectors are orthogonal → no shared vocabulary

Advantage: Length-normalized — a 3-page document does not automatically score
higher than a 1-page document simply because it contains more terms.
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
