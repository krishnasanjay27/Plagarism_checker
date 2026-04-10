"""
Visualization Generation
=========================
Generates academic-quality visualizations for the Mini Dolos dashboard.

All figures are exported as base64-encoded PNG strings embedded in JSON
responses — no static file serving required.

Produced Visualizations
------------------------
1. Similarity Heatmap
   An annotated n×n matrix (seaborn + matplotlib) showing pairwise cosine
   similarity scores. Uses the 'Blues' sequential colormap where white
   indicates zero similarity and dark blue indicates identical content.

2. Document Similarity Network Graph
   A node-link diagram (networkx + matplotlib) where:
     - Nodes represent documents (labeled with filename stems)
     - Edges connect document pairs whose similarity > threshold
     - Edge color encodes severity (red = very high, amber = high)
     - Edge thickness scales with similarity score
"""

import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — required on server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

# ── Shared Style Defaults ─────────────────────────────────────────────────────
TITLE_SIZE = 12
AXIS_LABEL_SIZE = 9
ANNOT_SIZE = 8

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _fig_to_base64(fig) -> str:
    """Serialize a matplotlib Figure to a base64 PNG string, then close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _shorten_label(name: str, max_len: int = 16) -> str:
    """Strip .txt extension and truncate long filenames for axis legibility."""
    stem = name.replace(".txt", "")
    return stem if len(stem) <= max_len else stem[:max_len - 2] + ".."


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def generate_heatmap(documents: list, matrix: list) -> str:
    """
    Generate an annotated cosine similarity heatmap.

    Colormap: 'Blues' sequential (white=0.0, dark blue=1.0)
    Diagonal: Grayed out (#F3F4F6) — self-similarity (always 1.0) is expected
              and visually distracts from cross-document comparisons.
    Annotations: Rounded to 2 decimal places inside each cell.

    Args:
        documents: Ordered list of document filenames.
        matrix:    n×n list of cosine similarity scores (floats in [0, 1]).

    Returns:
        Base64-encoded PNG string.
    """
    n = len(documents)
    labels = [_shorten_label(d) for d in documents]
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    # Dynamic figure size — scales with document count
    fig_w = max(6.5, n * 1.3)
    fig_h = max(5.5, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Mask the diagonal so the gray overlay below is visible
    mask = np.eye(n, dtype=bool)

    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        mask=mask,
        ax=ax,
        linewidths=0.6,
        linecolor="#E5E7EB",
        cbar_kws={"label": "Cosine Similarity Score", "shrink": 0.75},
        annot_kws={"size": ANNOT_SIZE},
    )

    # Draw diagonal cells manually in a neutral gray
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, color="#F3F4F6", zorder=2))
        ax.text(i + 0.5, i + 0.5, "1.00",
                ha="center", va="center",
                fontsize=ANNOT_SIZE, color="#9CA3AF")

    ax.set_title(
        "Document Similarity Matrix (Cosine Similarity)",
        fontsize=TITLE_SIZE, fontweight="bold", pad=14, color="#111827",
    )
    ax.set_xlabel("Documents", fontsize=AXIS_LABEL_SIZE + 1,
                  labelpad=10, color="#374151")
    ax.set_ylabel("Documents", fontsize=AXIS_LABEL_SIZE + 1,
                  labelpad=10, color="#374151")
    ax.tick_params(axis="both", labelsize=AXIS_LABEL_SIZE, colors="#4B5563")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Network Graph
# ─────────────────────────────────────────────────────────────────────────────

def generate_network(documents: list, suspicious_pairs: list) -> str:
    """
    Generate a document similarity network graph.

    Graph structure:
      - Nodes: one per document, labeled with the filename stem
      - Edges: drawn only between suspicious pairs (similarity > threshold)
      - Edge color:  Red  (#EF4444) for score ≥ 0.85 (very high)
                     Amber (#F59E0B) for score  0.75–0.84 (high)
      - Edge width:  Linearly scaled by score (thicker = more similar)
      - Edge labels: Cosine similarity score shown on each edge

    Layout: spring_layout with seed=42 for reproducibility.
    Falls back to circular_layout for ≤3 nodes to avoid degenerate layouts.

    Args:
        documents:        Ordered list of document filenames.
        suspicious_pairs: List of {'doc1', 'doc2', 'score'} dicts.

    Returns:
        Base64-encoded PNG string.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("white")

    G = nx.Graph()
    for doc in documents:
        G.add_node(doc)
    for pair in suspicious_pairs:
        G.add_edge(pair["doc1"], pair["doc2"],
                   weight=pair["score"], label=f"{pair['score']:.2f}")

    # Layout selection
    if len(documents) <= 3:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.5 / max(1, len(documents) ** 0.5))

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color="#1F2937",
        node_size=2400,
        ax=ax, alpha=0.93,
    )

    # Node labels
    node_labels = {doc: _shorten_label(doc) for doc in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=node_labels,
        font_size=8, font_color="white", font_weight="bold", ax=ax,
    )

    # Draw edges with score-based styling
    if G.edges():
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        edge_colors = [
            "#EF4444" if w >= 0.85 else "#F59E0B"
            for w in edge_weights
        ]
        edge_widths = [max(1.5, w * 5) for w in edge_weights]

        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.75,
            ax=ax,
        )

        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="#111827",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor="#D1D5DB"),
            ax=ax,
        )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#EF4444", label="Very high similarity  (≥ 0.85)"),
        mpatches.Patch(color="#F59E0B", label="High similarity  (0.75 – 0.84)"),
        mpatches.Patch(color="#1F2937", label="Document node"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        frameon=True,
        edgecolor="#E5E7EB",
    )

    ax.set_title(
        "Document Similarity Network Graph",
        fontsize=TITLE_SIZE, fontweight="bold", pad=14, color="#111827",
    )
    plt.axis("off")
    fig.tight_layout()
    return _fig_to_base64(fig)
