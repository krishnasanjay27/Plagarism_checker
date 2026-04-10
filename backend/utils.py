"""
Utils — Shared Helper Functions
================================
Utility functions for file metadata extraction, base64 image encoding,
and text snippet generation used across the Mini Dolos backend.
"""

import io
import re
import base64


def figure_to_base64(fig) -> str:
    """
    Convert a matplotlib Figure to a base64-encoded PNG string.

    Used to embed visualization images directly in JSON API responses,
    avoiding the need to serve static files from disk.

    Args:
        fig: A matplotlib Figure object.

    Returns:
        Base64-encoded PNG string (UTF-8 decoded).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def get_word_count(text: str) -> int:
    """
    Count the number of words in a text string.

    Splits on whitespace and filters empty tokens to handle
    multi-space separators and leading/trailing whitespace.

    Args:
        text: Raw document text.

    Returns:
        Integer word count.
    """
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def get_file_metadata(filename: str, content: str) -> dict:
    """
    Extract basic file metadata from a filename and its text content.

    Args:
        filename: Original uploaded filename (e.g., 'assignment1.txt').
        content:  File content as a decoded UTF-8 string.

    Returns:
        Dictionary with keys:
          - filename:   original filename
          - size_bytes: file size in bytes
          - size_kb:    file size in kilobytes (2 decimal places)
          - word_count: number of words in content
          - char_count: total character count
    """
    encoded_size = len(content.encode("utf-8"))
    return {
        "filename": filename,
        "size_bytes": encoded_size,
        "size_kb": round(encoded_size / 1024, 2),
        "word_count": get_word_count(content),
        "char_count": len(content),
    }


def get_text_snippet(text: str, max_chars: int = 200) -> str:
    """
    Return the first max_chars characters of a text, truncated at a word boundary.

    Used for generating preview snippets of uploaded documents in the
    preprocessing preview panel.

    Args:
        text:      Raw document text.
        max_chars: Maximum character count for the returned snippet.

    Returns:
        Truncated text ending with '...' if the original was longer.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    last_space = snippet.rfind(" ")
    if last_space > 0:
        snippet = snippet[:last_space]
    return snippet + "..."
