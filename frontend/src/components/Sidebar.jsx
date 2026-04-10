import React from "react";
import { NavLink } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";

const UploadIcon = () => (
  <svg viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
  </svg>
);
const ResultsIcon = () => (
  <svg viewBox="0 0 20 20" fill="currentColor">
    <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
    <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
  </svg>
);
const VizIcon = () => (
  <svg viewBox="0 0 20 20" fill="currentColor">
    <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
  </svg>
);

export default function Sidebar() {
  const { state } = useAnalysis();
  const docCount = state.fileMetadata.length;
  const hasResults = !!state.results;

  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <h1>Mini Dolos</h1>
        <p>Plagiarism Detection System</p>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-label">Workflow</div>

        <NavLink
          to="/"
          end
          className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
        >
          <UploadIcon />
          Upload Files
          {docCount > 0 && (
            <span style={{
              marginLeft: "auto",
              fontSize: "10px",
              background: "var(--accent-bg)",
              color: "var(--accent)",
              border: "1px solid #a7f3d0",
              borderRadius: "10px",
              padding: "1px 6px",
              fontWeight: 600,
            }}>
              {docCount}
            </span>
          )}
        </NavLink>

        <NavLink
          to="/results"
          className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
          style={{ opacity: hasResults ? 1 : 0.5, pointerEvents: hasResults ? "auto" : "none" }}
        >
          <ResultsIcon />
          Analysis Results
          {hasResults && state.results.suspicious_pairs.length > 0 && (
            <span style={{
              marginLeft: "auto",
              fontSize: "10px",
              background: "var(--danger-bg)",
              color: "var(--danger-text)",
              border: "1px solid var(--danger-border)",
              borderRadius: "10px",
              padding: "1px 6px",
              fontWeight: 600,
            }}>
              {state.results.suspicious_pairs.length}
            </span>
          )}
        </NavLink>

        <NavLink
          to="/visualizations"
          className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
          style={{ opacity: hasResults ? 1 : 0.5, pointerEvents: hasResults ? "auto" : "none" }}
        >
          <VizIcon />
          Visualizations
        </NavLink>

        <div className="nav-section-label" style={{ marginTop: 16 }}>Status</div>
        <div style={{ padding: "4px 16px" }}>
          <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.8 }}>
            <div>
              <span style={{ color: "var(--text-secondary)" }}>Documents: </span>
              <strong style={{ color: "var(--text)" }}>{docCount}</strong>
            </div>
            <div>
              <span style={{ color: "var(--text-secondary)" }}>Threshold: </span>
              <strong style={{ color: "var(--accent)" }}>
                {(state.threshold * 100).toFixed(0)}%
              </strong>
            </div>
            {hasResults && (
              <div>
                <span style={{ color: "var(--text-secondary)" }}>Flagged: </span>
                <strong style={{ color: "var(--danger-text)" }}>
                  {state.results.suspicious_pairs.length} pair
                  {state.results.suspicious_pairs.length !== 1 ? "s" : ""}
                </strong>
              </div>
            )}
          </div>
        </div>
      </nav>

      <div className="sidebar-footer">
        NLP Course Project · Cosine Similarity
      </div>
    </aside>
  );
}
