import React, { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import {
  analyzeDocuments,
  getResults,
  getHeatmap,
  getNetwork,
  getSentences,
  getProcessed,
  getExplanations,
} from "../services/api";

// ── Cell coloring ─────────────────────────────────────────────────────────────
function cellClass(score, isSelf) {
  if (isSelf) return "matrix-cell-self";
  if (score >= 0.90) return "matrix-cell-high";
  if (score >= 0.75) return "matrix-cell-warn";
  if (score >= 0.50) return "matrix-cell-mod";
  return "matrix-cell-low";
}

function scoreColor(score) {
  if (score >= 0.90) return "var(--danger-text)";
  if (score >= 0.75) return "var(--warn-text)";
  return "var(--accent)";
}

function classify(score) {
  if (score >= 0.90) return "Very high similarity";
  if (score >= 0.75) return "Potential plagiarism";
  if (score >= 0.50) return "Moderate similarity";
  return "Low similarity";
}

// ── CSV Export ────────────────────────────────────────────────────────────────
function exportCSV(documents, matrix) {
  const header = ["", ...documents].join(",");
  const rows = documents.map((doc, i) =>
    [doc, ...matrix[i].map((v) => (v * 100).toFixed(1) + "%")].join(",")
  );
  const csv = [header, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "similarity_matrix.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// ── Sub-components ────────────────────────────────────────────────────────────
function SummaryPanel({ results, threshold }) {
  if (!results) return null;
  const maxScore = results.suspicious_pairs.length > 0
    ? results.suspicious_pairs[0].score
    : null;
  return (
    <div className="metric-grid">
      <div className="metric-cell">
        <div className="metric-value">{results.documents.length}</div>
        <div className="metric-label">Documents Uploaded</div>
      </div>
      <div className="metric-cell">
        <div className={`metric-value ${results.suspicious_pairs.length > 0 ? "danger" : "accent"}`}>
          {results.suspicious_pairs.length}
        </div>
        <div className="metric-label">Suspicious Pairs</div>
      </div>
      <div className="metric-cell">
        <div className={`metric-value ${maxScore && maxScore >= 0.75 ? "warn" : "accent"}`}>
          {maxScore ? (maxScore * 100).toFixed(1) + "%" : "—"}
        </div>
        <div className="metric-label">Highest Similarity</div>
      </div>
      <div className="metric-cell">
        <div className="metric-value accent">{(threshold * 100).toFixed(0)}%</div>
        <div className="metric-label">Detection Threshold</div>
      </div>
    </div>
  );
}

function SimilarityLegend() {
  return (
    <div className="legend-row">
      <div className="legend-item">
        <div className="legend-dot" style={{ background: "var(--danger-bg)", border: "1px solid var(--danger-border)" }} />
        0.90–1.00 — Very high similarity
      </div>
      <div className="legend-item">
        <div className="legend-dot" style={{ background: "var(--warn-bg)", border: "1px solid var(--warn-border)" }} />
        0.75–0.89 — Potential plagiarism
      </div>
      <div className="legend-item">
        <div className="legend-dot" style={{ background: "var(--mod-bg)", border: "1px solid var(--mod-border)" }} />
        0.50–0.74 — Moderate similarity
      </div>
      <div className="legend-item">
        <div className="legend-dot" style={{ background: "var(--bg-subtle)", border: "1px solid var(--border)" }} />
        0.00–0.49 — Low similarity
      </div>
    </div>
  );
}

function MatrixTable({ documents, matrix }) {
  const shortName = (n) => n.replace(".txt", "");
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="matrix-table">
        <thead>
          <tr>
            <th style={{ textAlign: "left", minWidth: 120 }}>Document</th>
            {documents.map((d) => (
              <th key={d}>{shortName(d)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {documents.map((rowDoc, i) => (
            <tr key={rowDoc}>
              <td style={{
                textAlign: "left",
                fontWeight: 600,
                fontSize: 11,
                color: "var(--text-secondary)",
                background: "var(--bg-subtle)",
              }}>
                {shortName(rowDoc)}
              </td>
              {documents.map((colDoc, j) => {
                const score = matrix[i][j];
                const isSelf = i === j;
                const cls = cellClass(score, isSelf);
                return (
                  <td key={colDoc} className={cls} style={{ position: "relative" }}>
                    <div className="tooltip-wrap">
                      {isSelf ? "—" : `${(score * 100).toFixed(1)}%`}
                      {!isSelf && (
                        <div className="tooltip-box">
                          {shortName(rowDoc)} vs {shortName(colDoc)}<br />
                          {classify(score)}
                        </div>
                      )}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AlertCards({ pairs, explanations }) {
  if (!pairs || pairs.length === 0) {
    return (
      <div className="empty-state">
        <div>✓</div>
        <div style={{ marginTop: 6 }}>No suspicious pairs detected at current threshold.</div>
      </div>
    );
  }
  return (
    <div>
      {pairs.map((pair) => {
        const key = `${pair.doc1}|||${pair.doc2}`;
        const phrases = explanations?.[key] || [];
        const isHigh = pair.score >= 0.90;
        return (
          <div key={key} className={`alert-card ${isHigh ? "" : "warn"}`}>
            <div>
              <div className="alert-doc-label">Document A</div>
              <div className="alert-doc-name">{pair.doc1}</div>
            </div>
            <div>
              <div className="alert-doc-label">Document B</div>
              <div className="alert-doc-name">{pair.doc2}</div>
            </div>
            <div>
              <div className="alert-doc-label">Similarity</div>
              <div className="alert-score">{(pair.score * 100).toFixed(1)}%</div>
            </div>
            {phrases.length > 0 && (
              <div className="alert-phrases">
                <span style={{ fontSize: 10, color: "var(--text-muted)", marginRight: 4, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Shared phrases:
                </span>
                {phrases.slice(0, 6).map((p) => (
                  <span key={p} className="phrase-tag">{p}</span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RankingPanel({ pairs }) {
  if (!pairs || pairs.length === 0) {
    return <div className="empty-state">No suspicious pairs to rank.</div>;
  }
  return (
    <div>
      {pairs.map((pair, i) => (
        <div key={`${pair.doc1}|||${pair.doc2}`} className="rank-row">
          <div className="rank-num">{i + 1}</div>
          <div className="rank-docs">
            <span style={{ fontFamily: "monospace", fontSize: 12 }}>{pair.doc1}</span>
            <span style={{ color: "var(--text-muted)", margin: "0 6px" }}>vs</span>
            <span style={{ fontFamily: "monospace", fontSize: 12 }}>{pair.doc2}</span>
          </div>
          <div className="score-bar-wrap" style={{ width: 120 }}>
            <div className="score-bar-bg">
              <div
                className="score-bar-fill"
                style={{
                  width: `${pair.score * 100}%`,
                  background: pair.score >= 0.90 ? "#ef4444" : pair.score >= 0.75 ? "#f59e0b" : "var(--accent)",
                }}
              />
            </div>
          </div>
          <div className="rank-score" style={{ color: scoreColor(pair.score) }}>
            {(pair.score * 100).toFixed(1)}%
          </div>
        </div>
      ))}
    </div>
  );
}

function SentencePanel({ sentences }) {
  if (!sentences) return <div className="empty-state">Run analysis to see sentence matches.</div>;
  const keys = Object.keys(sentences);
  const allEmpty = keys.every((k) => !sentences[k] || sentences[k].length === 0);
  if (keys.length === 0 || allEmpty) {
    return <div className="empty-state">No significant sentence matches found.</div>;
  }
  return (
    <div>
      {keys.map((key) => {
        const matches = sentences[key];
        if (!matches || matches.length === 0) return null;
        const [d1, d2] = key.split("|||");
        return (
          <div key={key} style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: "var(--text-secondary)" }}>
              {d1} → {d2}
              <span style={{ fontWeight: 400, marginLeft: 8, color: "var(--text-muted)" }}>
                {matches.length} match{matches.length !== 1 ? "es" : ""}
              </span>
            </div>
            {matches.map((m, i) => (
              <div key={i} className="sentence-match">
                <div className="sentence-match-header">
                  <span>Match #{i + 1}</span>
                  <span style={{ color: scoreColor(m.score) }}>{(m.score * 100).toFixed(1)}%</span>
                </div>
                <div className="sentence-row">
                  <div className="sentence-cell">
                    <div className="sentence-cell-label">{d1}</div>
                    {m.sentence1}
                  </div>
                  <div className="sentence-cell">
                    <div className="sentence-cell-label">{d2}</div>
                    {m.sentence2}
                  </div>
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function ProcessingPreview({ processed }) {
  if (!processed) return <div className="empty-state">Run analysis to see preprocessing output.</div>;
  const keys = Object.keys(processed);
  return (
    <div>
      {keys.map((filename) => {
        const item = processed[filename];
        return (
          <div key={filename} style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              {filename}
              <span style={{ fontWeight: 400, marginLeft: 8, color: "var(--text-muted)" }}>
                {item.token_count} tokens
              </span>
            </div>
            <div className="preview-grid">
              <div className="preview-cell">
                <div className="preview-cell-label">Original (snippet)</div>
                <div className="preview-text">{item.original_snippet}</div>
              </div>
              <div className="preview-cell">
                <div className="preview-cell-label">Processed tokens</div>
                <div className="preview-text">{item.tokens}</div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SharedPhrasesPanel({ explanations }) {
  if (!explanations) return <div className="empty-state">Run analysis to see shared phrases.</div>;
  const keys = Object.keys(explanations);
  if (keys.length === 0) return (
    <div className="empty-state">No suspicious pairs — no shared phrase analysis available.</div>
  );
  return (
    <div>
      {keys.map((key) => {
        const phrases = explanations[key] || [];
        const [d1, d2] = key.split("|||");
        return (
          <div key={key} style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text-secondary)" }}>
              Shared phrases: <span style={{ color: "var(--text)" }}>{d1}</span>
              <span style={{ color: "var(--text-muted)", margin: "0 5px" }}>↔</span>
              <span style={{ color: "var(--text)" }}>{d2}</span>
            </div>
            {phrases.length === 0 ? (
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>No shared phrases found.</span>
            ) : (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {phrases.map((p) => (
                  <span key={p} className="phrase-tag" style={{ fontSize: 12, padding: "3px 9px" }}>{p}</span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
const TABS = [
  { id: "matrix",     label: "Similarity Matrix" },
  { id: "alerts",     label: "Plagiarism Alerts" },
  { id: "ranking",    label: "Similarity Ranking" },
  { id: "sentences",  label: "Sentence Matches" },
  { id: "phrases",    label: "Shared Phrases" },
  { id: "preprocess", label: "Preprocessing Preview" },
];

export default function ResultsPage() {
  const { state, dispatch } = useAnalysis();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("matrix");
  const [rerunning, setRerunning] = useState(false);

  const { results, explanations, sentences, processed, threshold } = state;

  // ── Re-run with new threshold ─────────────────────────────────────────────
  const rerunAnalysis = useCallback(async (newThreshold) => {
    setRerunning(true);
    try {
      await analyzeDocuments(newThreshold);
      const [resR, heatR, netR, sentR, procR, expR] = await Promise.all([
        getResults(),
        getHeatmap(),
        getNetwork(),
        getSentences().catch(() => ({ data: { matches: {} } })),
        getProcessed(),
        getExplanations().catch(() => ({ data: {} })),
      ]);
      dispatch({ type: "SET_RESULTS",      payload: resR.data });
      dispatch({ type: "SET_HEATMAP",      payload: heatR.data.image });
      dispatch({ type: "SET_NETWORK",      payload: netR.data.image });
      dispatch({ type: "SET_SENTENCES",    payload: sentR.data.matches });
      dispatch({ type: "SET_PROCESSED",    payload: procR.data });
      dispatch({ type: "SET_EXPLANATIONS", payload: expR.data });
    } catch (e) {
      console.error(e);
    }
    setRerunning(false);
  }, [dispatch]);

  // Debounce slider changes
  const [sliderTimer, setSliderTimer] = useState(null);
  const onThresholdChange = (val) => {
    dispatch({ type: "SET_THRESHOLD", payload: val });
    if (sliderTimer) clearTimeout(sliderTimer);
    const t = setTimeout(() => rerunAnalysis(val), 700);
    setSliderTimer(t);
  };

  if (!results) {
    return (
      <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
        <div className="page-header">
          <h2>Analysis Results</h2>
          <p>No results yet. Upload documents and run analysis first.</p>
        </div>
        <div className="page-content">
          <button className="btn btn-primary" onClick={() => navigate("/")}>
            ← Go to Upload
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <div className="page-header">
        <h2>Analysis Results</h2>
        <p>
          Plagiarism detection report — {results.documents.length} documents analyzed.
        </p>
      </div>

      <div className="page-content" style={{ flex: 1, overflowY: "auto" }}>

        {/* Summary Metrics */}
        <SummaryPanel results={results} threshold={threshold} />

        {/* Threshold Slider */}
        <div className="threshold-row">
          <span className="threshold-label">Detection Threshold</span>
          <input
            type="range"
            min="0.6" max="0.9" step="0.01"
            value={threshold}
            onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
            className="threshold-slider"
          />
          <span className="threshold-value">
            {rerunning ? <span className="spinner" /> : `${(threshold * 100).toFixed(0)}%`}
          </span>
        </div>

        {/* Tab Bar */}
        <div className="tab-bar">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-btn${activeTab === tab.id ? " active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
              {tab.id === "alerts" && results.suspicious_pairs.length > 0 && (
                <span style={{
                  marginLeft: 5,
                  background: "var(--danger-bg)",
                  color: "var(--danger-text)",
                  fontSize: 10,
                  fontWeight: 700,
                  border: "1px solid var(--danger-border)",
                  borderRadius: 8,
                  padding: "0 5px",
                }}>
                  {results.suspicious_pairs.length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === "matrix" && (
          <div className="panel">
            <div className="panel-header">
              <h3>Pairwise Similarity Matrix</h3>
              <button
                className="btn btn-sm"
                onClick={() => exportCSV(results.documents, results.matrix)}
              >
                ↓ Download CSV
              </button>
            </div>
            <div className="panel-body">
              <MatrixTable documents={results.documents} matrix={results.matrix} />
              <div className="divider" />
              <div className="section-title">Similarity Legend</div>
              <SimilarityLegend />
            </div>
          </div>
        )}

        {activeTab === "alerts" && (
          <div className="panel">
            <div className="panel-header">
              <h3>Plagiarism Alerts</h3>
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                Threshold: {(threshold * 100).toFixed(0)}%
              </span>
            </div>
            <div className="panel-body">
              <AlertCards pairs={results.suspicious_pairs} explanations={explanations} />
            </div>
          </div>
        )}

        {activeTab === "ranking" && (
          <div className="panel">
            <div className="panel-header">
              <h3>Similarity Ranking</h3>
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                Sorted by similarity score (highest first)
              </span>
            </div>
            <div className="panel-body">
              <RankingPanel pairs={results.suspicious_pairs} />
            </div>
          </div>
        )}

        {activeTab === "sentences" && (
          <div className="panel">
            <div className="panel-header">
              <h3>Sentence-Level Matches</h3>
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                Min. 5 words per sentence · Threshold ≥ 60%
              </span>
            </div>
            <div className="panel-body">
              <SentencePanel sentences={sentences} />
            </div>
          </div>
        )}

        {activeTab === "phrases" && (
          <div className="panel">
            <div className="panel-header">
              <h3>Shared Key Phrases</h3>
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                Top n-grams contributing to similarity
              </span>
            </div>
            <div className="panel-body">
              <SharedPhrasesPanel explanations={explanations} />
            </div>
          </div>
        )}

        {activeTab === "preprocess" && (
          <div className="panel">
            <div className="panel-header">
              <h3>Preprocessing Preview</h3>
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                Original snippet vs processed tokens
              </span>
            </div>
            <div className="panel-body">
              <ProcessingPreview processed={processed} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
