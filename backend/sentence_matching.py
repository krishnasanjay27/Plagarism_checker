"""
Sentence-Level Similarity Matching
=====================================
Provides fine-grained plagiarism evidence by detecting matching or
near-identical sentences between suspicious document pairs.

Why sentence-level matching matters
-------------------------------------
Document-level cosine similarity produces a single global score, but it does
not reveal *which* sentences were copied or paraphrased. Sentence-level
matching surfaces the exact textual evidence needed for academic integrity
review, making the tool's findings auditable and explainable.

Method
-------
  1. Split each document into sentences using NLTK's Punkt tokenizer.
  2. Filter sentences shorter than `min_words` to suppress false positives
     from generic short phrases (e.g., "This paper discusses.", "In conclusion.").
  3. Preprocess each sentence through the same pipeline as documents
     (lowercase → remove punctuation → remove stopwords → lemmatize).
  4. Fit a shared TF-IDF vocabulary over all sentences from both documents.
  5. Compute an m×k cosine similarity matrix (m sentences from doc A,
     k sentences from doc B).
  6. Collect pairs with similarity ≥ sim_threshold, sort by score, return top_k.

Noise-Reduction Rule — Minimum Word Filter
--------------------------------------------
Short sentences inflate sentence-level match counts without providing
meaningful plagiarism evidence. The 5-word minimum threshold (`min_words=5`)
is chosen to exclude:
    "Introduction."
    "See Table 1."
    "The results show."
while still capturing the genuine sentence-level matches that matter.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import get_sentences, preprocess


def match_sentences(
    documents: dict,
    suspicious_pairs: list,
    min_words: int = 5,
    sim_threshold: float = 0.6,
    top_k: int = 10,
) -> dict:
    """
    Find matching or near-identical sentences between suspicious document pairs.

    Args:
        documents:        {filename: raw_text} — original unprocessed documents.
        suspicious_pairs: List of {'doc1', 'doc2', 'score'} dicts from
                          similarity.compute_similarity().
        min_words:        Minimum word count for a sentence to be included.
                          Sentences with fewer words are skipped to reduce
                          false positives from generic short phrases.
                          Default: 5 words.
        sim_threshold:    Minimum cosine similarity for a sentence pair to be
                          reported as a match. Default: 0.6 (60% similarity).
        top_k:            Maximum number of sentence matches to return per pair.
                          Default: 10.

    Returns:
        {
          'docA|||docB': [
            {
              'sentence1': 'Original sentence from document A...',
              'sentence2': 'Matching sentence from document B...',
              'score':      0.87
            },
            ...
          ],
          ...
        }
        Keys use '|||' as delimiter to match the format used in similarity.py.
    """
    matches = {}

    for pair in suspicious_pairs:
        doc1_name = pair["doc1"]
        doc2_name = pair["doc2"]
        key = f"{doc1_name}|||{doc2_name}"

        # Split documents into raw sentences
        raw_sents1 = get_sentences(documents.get(doc1_name, ""))
        raw_sents2 = get_sentences(documents.get(doc2_name, ""))

        # Apply minimum word count filter
        # Filters out short generic phrases that produce noisy false positives
        filtered1 = [s for s in raw_sents1 if len(s.split()) >= min_words]
        filtered2 = [s for s in raw_sents2 if len(s.split()) >= min_words]

        if not filtered1 or not filtered2:
            matches[key] = []
            continue

        # Preprocess each sentence through the same pipeline as documents
        processed1 = [preprocess(s) for s in filtered1]
        processed2 = [preprocess(s) for s in filtered2]

        # Build a shared TF-IDF vocabulary from all sentences in both documents
        # — ensures the feature space is consistent across both sets
        all_processed = processed1 + processed2
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            all_vectors = vectorizer.fit_transform(all_processed)
        except ValueError:
            # If all sentences produce empty vocabulary (e.g., only stopwords)
            matches[key] = []
            continue

        n1 = len(processed1)
        vectors1 = all_vectors[:n1]       # Sentence vectors from doc A
        vectors2 = all_vectors[n1:]       # Sentence vectors from doc B

        # Compute m×k cosine similarity matrix between all sentence pairs
        sim_matrix = cosine_similarity(vectors1, vectors2)

        # Collect sentence pairs above the threshold
        pair_matches = []
        for i in range(len(filtered1)):
            for j in range(len(filtered2)):
                score = float(sim_matrix[i][j])
                if score >= sim_threshold:
                    pair_matches.append({
                        "sentence1": filtered1[i].strip(),
                        "sentence2": filtered2[j].strip(),
                        "score": round(score, 4),
                    })

        # Sort by similarity score descending, return top_k results
        pair_matches.sort(key=lambda x: x["score"], reverse=True)
        matches[key] = pair_matches[:top_k]

    return matches
