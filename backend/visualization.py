"""
Visualization Generation
=========================
Generates academic-quality visualizations for the Mini Dolos dashboard.

All figures are exported as base64-encoded PNG strings embedded in JSON
responses — no static file serving required.

Produces Visualizations
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

3. TF-IDF Vector Space (PCA Projection)
   A 2D scatter plot produced by reducing high-dimensional TF-IDF vectors
   with Principal Component Analysis (PCA). Documents close together in
   this plot share similar term-weight distributions, which directly
   explains why they receive high cosine similarity scores.
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
from sklearn.decomposition import PCA

# ── Shared Style Defaults ─────────────────────────────────────────────────────
TITLE_SIZE = 12
AXIS_LABEL_SIZE = 9
ANNOT_SIZE = 8

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Cluster Colour Palettes ───────────────────────────────────────────────────
# Node fill colours (darker, saturated — visible against white background)
_CLUSTER_NODE_COLORS = [
    "#065F46",   # dark green
    "#1E40AF",   # dark blue
    "#7C2D12",   # dark red-brown
    "#4C1D95",   # dark purple
    "#0F766E",   # teal
    "#92400E",   # amber-brown
    "#831843",   # maroon
    "#1E3A5F",   # navy
]
# Background patch colours (light, desaturated — for the polygon fill)
_CLUSTER_BG_COLORS = [
    "#D1FAE5",   # mint green
    "#DBEAFE",   # pale blue
    "#FEE2E2",   # pale red
    "#EDE9FE",   # pale purple
    "#CCFBF1",   # pale teal
    "#FEF3C7",   # pale amber
    "#FCE7F3",   # pale pink
    "#EFF6FF",   # very pale blue
]


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


# ─────────────────────────────────────────────────────────────────────────────
# PCA Vector Space Projection
# ─────────────────────────────────────────────────────────────────────────────

def generate_pca_plot(documents: list, tfidf_matrix, suspicious_pairs: list) -> str:
    """
    Project TF-IDF document vectors into 2D space using PCA and render as scatter plot.

    Mathematical basis:
      Each document is a vector in R^V where V = vocabulary size (can be thousands).
      PCA finds the two orthogonal axes (principal components) that capture the
      maximum variance across all document vectors. Projecting onto these axes
      preserves the relative distances between documents as faithfully as possible.

    Visual interpretation:
      - Documents clustered together share similar TF-IDF term distributions.
      - Documents far apart have very different vocabularies.
      - This directly visualises why cosine similarity scores are high or low:
        documents whose vectors point in the same direction will cluster together.

    Edge annotations:
      Suspicious document pairs (similarity above threshold) are connected by
      colored annotation lines to make the relationship explicit.

    Layout:
      Axes are labelled with their explained variance percentage so the
      reader knows how much information is captured in each dimension.

    Args:
        documents:        Ordered list of document filenames.
        tfidf_matrix:     Sparse TF-IDF matrix from compute_similarity().
        suspicious_pairs: List of {doc1, doc2, score} for edge annotation.

    Returns:
        Base64-encoded PNG string.
    """
    dense  = tfidf_matrix.toarray()
    n      = len(documents)
    labels = [_shorten_label(d) for d in documents]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_facecolor("white")

    # ── Handle degenerate cases ──────────────────────────────────────────────
    if n < 2:
        ax.text(0.5, 0.5, "Need ≥ 2 documents for PCA projection.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12,
                color="#9CA3AF")
        ax.axis("off")
        return _fig_to_base64(fig)

    # ── PCA reduction ────────────────────────────────────────────────────
    # n_components must not exceed min(n_samples, n_features)
    n_components = min(2, n, dense.shape[1])
    pca = PCA(n_components=n_components, random_state=42)

    try:
        reduced = pca.fit_transform(dense)
    except Exception as e:
        ax.text(0.5, 0.5, f"PCA failed: {str(e)}",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _fig_to_base64(fig)

    ev = pca.explained_variance_ratio_

    if n_components == 1:
        # Only 1 component: plot on x-axis with y=0
        x_coords = reduced[:, 0]
        y_coords = np.zeros(n)
        xlabel = f"PC1 ({ev[0] * 100:.1f}% variance explained)"
        ylabel = ""
    else:
        x_coords = reduced[:, 0]
        y_coords = reduced[:, 1]
        xlabel = f"PC1 ({ev[0] * 100:.1f}% variance)"
        ylabel = f"PC2 ({ev[1] * 100:.1f}% variance)"

    # ── Draw suspicious pair connections ──────────────────────────────────────
    doc_idx = {doc: i for i, doc in enumerate(documents)}
    for pair in suspicious_pairs:
        i1 = doc_idx.get(pair["doc1"])
        i2 = doc_idx.get(pair["doc2"])
        if i1 is not None and i2 is not None:
            score  = pair["score"]
            color  = "#EF4444" if score >= 0.85 else "#F59E0B"
            alpha  = min(0.9, score)  # Higher similarity = more opaque line
            ax.plot(
                [x_coords[i1], x_coords[i2]],
                [y_coords[i1], y_coords[i2]],
                color=color, alpha=alpha, linewidth=1.5, zorder=1,
                linestyle="--",
            )
            # Score label at midpoint
            mx = (x_coords[i1] + x_coords[i2]) / 2
            my = (y_coords[i1] + y_coords[i2]) / 2
            ax.text(mx, my, f"{score:.2f}",
                    fontsize=7, ha="center", va="center",
                    color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor=color, alpha=0.85))

    # ── Draw document nodes and labels ───────────────────────────────────────
    # Colour suspicious documents differently
    suspicious_docs = set()
    for p in suspicious_pairs:
        suspicious_docs.add(p["doc1"])
        suspicious_docs.add(p["doc2"])

    node_colors = ["#991B1B" if d in suspicious_docs else "#1F2937" for d in documents]

    ax.scatter(x_coords, y_coords, c=node_colors, s=200, zorder=5, alpha=0.92)

    for i, label in enumerate(labels):
        ax.annotate(
            label, (x_coords[i], y_coords[i]),
            textcoords="offset points", xytext=(10, 5),
            fontsize=9, color="#374151", fontweight="500",
        )

    # ── Axes, title, legend ───────────────────────────────────────────────────
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE + 1, color="#4B5563", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE + 1, color="#4B5563", labelpad=8)
    ax.set_title(
        "TF-IDF Document Vector Space (PCA 2D Projection)",
        fontsize=TITLE_SIZE, fontweight="bold", pad=14, color="#111827",
    )
    ax.grid(True, color="#F3F4F6", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Variance info
    total_var = sum(ev[:n_components]) * 100
    ax.text(0.01, 0.01, f"Total variance explained: {total_var:.1f}%",
            transform=ax.transAxes, fontsize=8, color="#9CA3AF")

    # Legend handles
    legend_h = [
        mpatches.Patch(color="#991B1B", label="Flagged document"),
        mpatches.Patch(color="#1F2937", label="Clear document"),
    ]
    if suspicious_pairs:
        legend_h += [
            plt.Line2D([0], [0], color="#EF4444", linestyle="--",
                       linewidth=1.5, label="High similarity edge"),
        ]
    ax.legend(handles=legend_h, loc="upper right", fontsize=8,
              frameon=True, edgecolor="#E5E7EB")

    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Clustered Network Graph  (connected-component community detection)
# ─────────────────────────────────────────────────────────────────────────────

def generate_clustered_network(
    documents: list,
    matrix: list,
    threshold: float = 0.75,
    hide_singletons: bool = False,
) -> str:
    """
    Generate a color-coded document similarity network with cluster detection.

    Clustering method — Connected Components (networkx):
      At a given similarity threshold we build a graph where an edge (doc_i, doc_j)
      exists iff cosine_similarity(i, j) > threshold.  A connected component in
      this graph is a cluster of documents that are, directly or indirectly, similar
      enough to each other to be flagged. This mirrors Dolos-style cluster detection
      and is equivalent to single-linkage agglomerative clustering at this threshold.

    Visual encoding:
      - Each cluster gets a unique colour from _CLUSTER_NODE_COLORS.
      - A semi-transparent rounded FancyBboxPatch is drawn behind each cluster.
      - Singletons (no edge at this threshold) are hollow gray nodes.
        They can be hidden via hide_singletons=True.
      - Red edges: cosine >= 0.85  (very high),  Amber: threshold < score < 0.85.
      - Edge thickness scales linearly with cosine similarity score.

    Args:
        documents:       Ordered list of filenames.
        matrix:          n x n similarity matrix (serialised list-of-lists).
        threshold:       Minimum score to show an edge.
        hide_singletons: Remove isolated nodes when True.

    Returns:
        Base64-encoded PNG string.
    """
    n = len(documents)

    # ── Build graph ───────────────────────────────────────────────────────────
    G = nx.Graph()
    for doc in documents:
        G.add_node(doc)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(matrix[i][j])
            if score > threshold:
                G.add_edge(documents[i], documents[j], weight=score)

    # ── Connected components = clusters ───────────────────────────────────────
    components  = sorted(nx.connected_components(G), key=len, reverse=True)
    clusters    = [c for c in components if len(c) > 1]
    singletons  = [list(c)[0] for c in components if len(c) == 1]

    if hide_singletons:
        for s in singletons:
            G.remove_node(s)
        singletons = []

    node_cluster: dict = {}
    for ci, comp in enumerate(clusters):
        for node in comp:
            node_cluster[node] = ci
    for s in singletons:
        node_cluster[s] = -1

    # ── Empty-graph guard ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor("white")

    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5,
                "No documents to display.\nAll documents are singletons\n"
                "and Hide Singletons is enabled.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#9CA3AF", linespacing=1.8)
        ax.axis("off")
        return _fig_to_base64(fig)

    # ── Layout ────────────────────────────────────────────────────────────────
    if len(G.nodes()) <= 3:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(
            G, seed=42, k=3.0 / max(1, len(G.nodes()) ** 0.5)
        )

    # ── Cluster background patches ────────────────────────────────────────────
    for ci, comp in enumerate(clusters):
        active = [nd for nd in comp if nd in G.nodes()]
        if not active:
            continue
        fg  = _CLUSTER_NODE_COLORS[ci % len(_CLUSTER_NODE_COLORS)]
        bg  = _CLUSTER_BG_COLORS[ci  % len(_CLUSTER_BG_COLORS)]
        xs  = [pos[nd][0] for nd in active]
        ys  = [pos[nd][1] for nd in active]
        margin = 0.22
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        w  = max(max(xs) - min(xs), 0.30) + 2 * margin
        h  = max(max(ys) - min(ys), 0.30) + 2 * margin
        patch = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.8, edgecolor=fg, facecolor=bg,
            alpha=0.38, zorder=0,
        )
        ax.add_patch(patch)
        # Cluster label badge
        ax.text(
            cx, cy + h / 2 + 0.04,
            f"Cluster {ci + 1}  ({len(active)} doc{'s' if len(active) != 1 else ''})",
            ha="center", va="bottom", fontsize=8, color=fg, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=fg, alpha=0.85,
                      boxstyle="round,pad=0.25", linewidth=1),
            zorder=3,
        )

    # ── Edges ─────────────────────────────────────────────────────────────────
    if G.edges():
        ew = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            edge_color=["#EF4444" if w >= 0.85 else "#F59E0B" for w in ew],
            width=[max(1.5, w * 5) for w in ew],
            alpha=0.70, ax=ax,
        )
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()},
            font_size=8, font_color="#111827",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#D1D5DB"),
            ax=ax,
        )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for node in G.nodes():
        ci   = node_cluster.get(node, -1)
        x, y = pos[node]
        if ci == -1:
            ax.scatter([x], [y], s=2400, c="white", edgecolors="#9CA3AF",
                       linewidths=2.5, zorder=5, alpha=0.95)
        else:
            color = _CLUSTER_NODE_COLORS[ci % len(_CLUSTER_NODE_COLORS)]
            ax.scatter([x], [y], s=2400, c=color, zorder=5,
                       alpha=0.92, edgecolors="white", linewidths=1.8)

    # ── Node labels ───────────────────────────────────────────────────────────
    for node in G.nodes():
        ci   = node_cluster.get(node, -1)
        x, y = pos[node]
        ax.text(x, y, _shorten_label(node),
                ha="center", va="center", fontsize=8,
                color="white" if ci != -1 else "#4B5563",
                fontweight="bold", zorder=6)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = []
    for ci, comp in enumerate(clusters):
        active = [nd for nd in comp if nd in G.nodes()]
        if not active:
            continue
        handles.append(mpatches.Patch(
            color=_CLUSTER_NODE_COLORS[ci % len(_CLUSTER_NODE_COLORS)],
            label=f"Cluster {ci + 1}  ({len(active)} document{'s' if len(active) != 1 else ''})",
        ))
    if singletons:
        handles.append(mpatches.Patch(
            facecolor="white", edgecolor="#9CA3AF", linewidth=2,
            label=f"Singleton  ({len(singletons)} isolated doc{'s' if len(singletons) != 1 else ''})",
        ))
    handles += [
        plt.Line2D([0], [0], color="#EF4444", linewidth=2.5,
                   label="Very high similarity  (>= 0.85)"),
        plt.Line2D([0], [0], color="#F59E0B", linewidth=2.5,
                   label=f"High similarity  ({threshold:.2f} to 0.84)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              frameon=True, edgecolor="#E5E7EB", facecolor="white")

    # ── Stats annotation + title ──────────────────────────────────────────────
    nc = len([c for c in clusters if any(nd in G.nodes() for nd in c)])
    ns = len(singletons)
    ax.text(0.5, 1.015,
            f"Threshold: {threshold:.0%}  "
            f"  {nc} cluster{'s' if nc != 1 else ''}  "
            f"  {ns} isolated doc{'s' if ns != 1 else ''}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#6B7280")
    ax.set_title(
        "Document Similarity Clusters  (Connected-Component Community Detection)",
        fontsize=TITLE_SIZE, fontweight="bold", pad=22, color="#111827",
    )
    plt.axis("off")
    fig.tight_layout()
    return _fig_to_base64(fig)
