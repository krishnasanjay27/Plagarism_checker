import React, { useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import PipelineStatus, { STEPS } from "../components/PipelineStatus";
import {
  uploadFiles,
  analyzeDocuments,
  getResults,
  getHeatmap,
  getNetwork,
  getSentences,
  getProcessed,
  getExplanations,
  getVectors,
  getTopTerms,
  getVectorSpace,
} from "../services/api";

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
};

function countWords(text) {
  return text ? text.trim().split(/\s+/).filter(Boolean).length : 0;
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function UploadPage() {
  const { state, dispatch } = useAnalysis();
  const navigate = useNavigate();
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  // Local file list (File objects) with preview metadata
  const [localFiles, setLocalFiles] = useState([]);

  const addFiles = useCallback((incoming) => {
    const txtFiles = Array.from(incoming).filter((f) =>
      f.name.toLowerCase().endsWith(".txt")
    );
    setLocalFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      const added = txtFiles.filter((f) => !names.has(f.name));
      return [...prev, ...added];
    });
    dispatch({ type: "RESET_ANALYSIS" });
    dispatch({ type: "SET_ERROR", payload: null });
  }, [dispatch]);

  const removeFile = (name) =>
    setLocalFiles((prev) => prev.filter((f) => f.name !== name));

  // Drag events
  const onDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = ()  => setDragging(false);
  const onDrop      = (e) => {
    e.preventDefault(); setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  // ── Run Analysis ─────────────────────────────────────────────────────────────
  const runAnalysis = async () => {
    if (localFiles.length < 2) {
      dispatch({ type: "SET_ERROR", payload: "Please upload at least 2 .txt files." });
      return;
    }

    const steps = STEPS.map((label) => ({ label, status: "pending" }));
    dispatch({ type: "SET_PIPELINE", payload: steps });
    dispatch({ type: "SET_ANALYZING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });

    const markActive = (i) => dispatch({ type: "STEP_ACTIVE", payload: i });
    const markDone   = (i) => dispatch({ type: "STEP_DONE",   payload: i });

    try {
      // Step 0 — Upload
      markActive(0);
      const uploadRes = await uploadFiles(localFiles);
      dispatch({ type: "SET_FILE_METADATA", payload: uploadRes.data.files });
      dispatch({ type: "SET_FILES", payload: localFiles });
      markDone(0);

      // Step 1 — Preprocess (fires inside /analyze)
      markActive(1);
      await new Promise((r) => setTimeout(r, 300));
      markDone(1);

      // Step 2 — TF-IDF
      markActive(2);
      await new Promise((r) => setTimeout(r, 300));
      markDone(2);

      // Step 3 — Cosine similarity
      markActive(3);
      await analyzeDocuments(state.threshold);
      markDone(3);

      // Step 4 — Visualizations & fetch all results
      markActive(4);
      const [resR, heatR, netR, sentR, procR, expR, vecR, topR, vsR] = await Promise.all([
        getResults(),
        getHeatmap(),
        getNetwork(),
        getSentences().catch(() => ({ data: { matches: {} } })),
        getProcessed(),
        getExplanations().catch(() => ({ data: {} })),
        getVectors().catch(() => ({ data: { terms: [], vectors: {}, total_features: 0, shown_features: 0 } })),
        getTopTerms().catch(() => ({ data: {} })),
        getVectorSpace().catch(() => ({ data: { image: null } })),
      ]);

      dispatch({ type: "SET_RESULTS",      payload: resR.data });
      dispatch({ type: "SET_HEATMAP",      payload: heatR.data.image });
      dispatch({ type: "SET_NETWORK",      payload: netR.data.image });
      dispatch({ type: "SET_SENTENCES",    payload: sentR.data.matches });
      dispatch({ type: "SET_PROCESSED",    payload: procR.data });
      dispatch({ type: "SET_EXPLANATIONS", payload: expR.data });
      dispatch({ type: "SET_VECTORS",      payload: vecR.data });
      dispatch({ type: "SET_TOP_TERMS",    payload: topR.data });
      dispatch({ type: "SET_VECTOR_SPACE", payload: vsR.data.image });
      markDone(4);

      dispatch({ type: "SET_ANALYZING", payload: false });

      // Navigate to results after a short delay so user sees completed steps
      setTimeout(() => navigate("/results"), 600);
    } catch (err) {
      const msg =
        err.response?.data?.error ||
        err.response?.data?.details ||
        err.message ||
        "Analysis failed. Is the Flask server running on port 5000?";
      dispatch({ type: "SET_ERROR", payload: msg });
      dispatch({ type: "SET_ANALYZING", payload: false });
    }
  };

  const isAnalyzing = state.isAnalyzing;
  const canAnalyze  = localFiles.length >= 2 && !isAnalyzing;

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Page Header */}
      <div className="page-header">
        <h2>Upload Documents</h2>
        <p>
          Upload multiple student assignment TXT files to detect plagiarism using
          TF-IDF vectorization and cosine similarity.
        </p>
      </div>

      <div className="page-content" style={{ flex: 1, overflowY: "auto" }}>

        {/* ── Drop Zone ─────────────────────────────────────────────────── */}
        <div className="panel">
          <div className="panel-header">
            <h3>File Upload</h3>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              .txt files only
            </span>
          </div>
          <div className="panel-body">
            <div
              className={`dropzone${dragging ? " dragging" : ""}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                multiple
                accept=".txt"
                onChange={(e) => addFiles(e.target.files)}
              />
              <div className="dropzone-icon">📄</div>
              <h3>Drag &amp; drop TXT files here</h3>
              <p>or click to browse — multiple files supported</p>
              <p style={{ marginTop: 6, fontSize: 11 }}>
                Supported format: <strong>.txt</strong> only
              </p>
            </div>
          </div>
        </div>

        {/* ── File Metadata Table ────────────────────────────────────────── */}
        {localFiles.length > 0 && (
          <div className="panel">
            <div className="panel-header">
              <h3>Selected Files</h3>
              <button
                className="btn btn-sm"
                onClick={() => setLocalFiles([])}
              >
                Clear all
              </button>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Filename</th>
                    <th>Size</th>
                    <th>Words (est.)</th>
                    <th>Type</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {localFiles.map((f, i) => (
                    <tr key={f.name}>
                      <td style={{ color: "var(--text-muted)", width: 32 }}>{i + 1}</td>
                      <td>
                        <span style={{ fontFamily: "Courier New, monospace", fontSize: 12 }}>
                          {f.name}
                        </span>
                      </td>
                      <td>{fmt(f.size)}</td>
                      <td style={{ color: "var(--text-secondary)" }}>
                        ~{Math.round(f.size / 5)}
                      </td>
                      <td>
                        <span className="badge badge-ok">TXT</span>
                      </td>
                      <td style={{ textAlign: "right" }}>
                        <button
                          className="btn btn-sm"
                          onClick={(e) => { e.stopPropagation(); removeFile(f.name); }}
                          style={{ padding: "2px 8px", color: "var(--danger-text)" }}
                        >
                          ✕
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Threshold + Analyze ────────────────────────────────────────── */}
        <div className="panel">
          <div className="panel-header">
            <h3>Analysis Configuration</h3>
          </div>
          <div className="panel-body">
            <div className="threshold-row">
              <span className="threshold-label">Similarity Threshold</span>
              <input
                type="range"
                min="0.6" max="0.9" step="0.01"
                value={state.threshold}
                onChange={(e) =>
                  dispatch({ type: "SET_THRESHOLD", payload: parseFloat(e.target.value) })
                }
                className="threshold-slider"
              />
              <span className="threshold-value">
                {(state.threshold * 100).toFixed(0)}%
              </span>
            </div>
            <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: "0 0 14px" }}>
              Document pairs with cosine similarity above this threshold will be flagged
              as potential plagiarism. Acceptable range: 60% – 90%.
            </p>

            {localFiles.length < 2 && (
              <p style={{ fontSize: 12, color: "var(--warn-text)", marginBottom: 12 }}>
                ⚠ Upload at least 2 files to run analysis.
              </p>
            )}

            {state.error && (
              <div style={{
                padding: "10px 14px",
                background: "var(--danger-bg)",
                border: "1px solid var(--danger-border)",
                borderRadius: 3,
                fontSize: 13,
                color: "var(--danger-text)",
                marginBottom: 12,
              }}>
                {state.error}
              </div>
            )}

            <button
              className="btn btn-primary"
              onClick={runAnalysis}
              disabled={!canAnalyze}
            >
              {isAnalyzing && <span className="spinner" />}
              {isAnalyzing ? "Analyzing…" : "Run Plagiarism Analysis"}
            </button>
          </div>
        </div>

        {/* ── Pipeline Status ────────────────────────────────────────────── */}
        {state.pipelineSteps.length > 0 && (
          <div className="panel">
            <div className="panel-header">
              <h3>Processing Pipeline</h3>
            </div>
            <div className="panel-body" style={{ padding: 0 }}>
              <PipelineStatus steps={state.pipelineSteps} />
            </div>
          </div>
        )}

        {/* ── Instructions ──────────────────────────────────────────────── */}
        {localFiles.length === 0 && (
          <div className="panel">
            <div className="panel-header"><h3>Instructions</h3></div>
            <div className="panel-body">
              <ol style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: "var(--text-secondary)", lineHeight: 2 }}>
                <li>Upload two or more student assignment <code>.txt</code> files above.</li>
                <li>Adjust the similarity threshold (default 75%) as needed.</li>
                <li>Click <strong>Run Plagiarism Analysis</strong>.</li>
                <li>Review the similarity matrix, flagged pairs, and visualizations.</li>
              </ol>
              <div className="divider" />
              <p style={{ fontSize: 12, color: "var(--text-muted)", margin: 0 }}>
                Sample data available in <code>sample_data/</code> directory —
                includes 5 assignments with varying similarity levels for testing.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
