"""
NLP Preprocessing Pipeline
===========================
Text normalization and cleaning module for the Mini Dolos plagiarism detection
system. Applied to all uploaded documents before TF-IDF vectorization.

Academic Context
-----------------
Raw student submissions contain surface-level noise that interferes with
content-level similarity measurement:
  - Capitalization differences ("Machine" vs "machine")
  - Punctuation variants
  - Common function words ("the", "is", "and") that appear everywhere
  - Inflectional variants ("running", "runs", "ran")

By normalizing all documents through a consistent preprocessing pipeline,
we ensure that semantically equivalent passages produce similar TF-IDF vectors
regardless of stylistic or grammatical variations, improving detection accuracy.

Pipeline Steps
--------------
  1. Lowercase normalization
  2. Punctuation removal
  3. Tokenization (NLTK Penn Treebank tokenizer)
  4. Stopword removal (NLTK English stopwords)
  5. Lemmatization (NLTK WordNet lemmatizer)
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# ── NLTK Resource Downloads ───────────────────────────────────────────────────
# Downloaded once and cached in the user's NLTK data directory.
# quiet=True suppresses download progress output.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Module-Level Singletons ───────────────────────────────────────────────────
# Instantiated once to avoid repeated initialization overhead during batch
# processing of multiple documents.
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
# Core Preprocessing Function
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Apply the full NLP preprocessing pipeline to a single document.

    Step 1 — Lowercase Normalization:
        Converts all characters to lowercase so that "Machine" and "machine"
        are treated as the same token. Prevents false negatives caused by
        capitalization differences in otherwise identical passages.

    Step 2 — Punctuation Removal:
        Removes all punctuation using Python's str.translate() with the
        full string.punctuation character set. Punctuation carries no
        semantic content for similarity purposes and inflates the vocabulary.

    Step 3 — Tokenization:
        Splits the cleaned string into individual word tokens using NLTK's
        word_tokenize() (Penn Treebank tokenizer). This handles edge cases
        like contractions ("don't" → "do", "n't") better than str.split().

    Step 4 — Stopword Removal:
        Removes English function words (the, is, are, a, an, of, in, …)
        that appear with high frequency across virtually all documents.
        These words would dominate TF-IDF features without contributing
        discriminative information for plagiarism detection.
        Only alphabetic tokens are retained to eliminate numeric noise.

    Step 5 — Lemmatization:
        Reduces each token to its canonical base form (lemma) using the
        WordNet lexical database:
            "running" → "run"    "algorithms" → "algorithm"
            "learned"  → "learn"  "studies"    → "study"
        Ensures different morphological forms of the same concept are counted
        together, improving recall for paraphrased plagiarism.

    Args:
        text: Raw document text as a string.

    Returns:
        Cleaned, space-separated string of lemmatized content words,
        ready for TF-IDF vectorization.
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Tokenize
    tokens = word_tokenize(text)

    # Step 4: Remove stopwords and non-alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and t not in _stop_words]

    # Step 5: Lemmatize to base form
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_all(documents: dict) -> dict:
    """
    Apply the preprocessing pipeline to all documents in the corpus.

    Args:
        documents: {filename: raw_text} mapping from the upload store.

    Returns:
        {filename: cleaned_text} mapping, same key order as input.
    """
    return {name: preprocess(text) for name, text in documents.items()}


def get_sentences(text: str) -> list:
    """
    Split a raw document into individual sentences using NLTK's
    Punkt sentence tokenizer.

    Used by the sentence-level matching module to identify
    matching or near-identical sentences between suspicious pairs.

    Args:
        text: Raw document text.

    Returns:
        List of sentence strings (original, un-preprocessed).
    """
    return sent_tokenize(text)
