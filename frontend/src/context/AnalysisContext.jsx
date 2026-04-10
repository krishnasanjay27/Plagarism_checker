import React, { createContext, useContext, useReducer } from "react";

// ── Initial State ─────────────────────────────────────────────────────────────
const initialState = {
  uploadedFiles:   [],     // File objects from <input>
  fileMetadata:    [],     // [{filename, size_kb, word_count}] from API
  threshold:       0.75,
  isUploading:     false,
  isAnalyzing:     false,
  pipelineSteps:   [],     // [{label, status: 'pending'|'active'|'done'}]
  results:         null,   // {documents, matrix, suspicious_pairs, threshold, metadata}
  explanations:    null,   // {docA|||docB: [phrases]}
  sentences:       null,   // {docA|||docB: [{sentence1, sentence2, score}]}
  processed:       null,   // {filename: {tokens, token_count, original_snippet}}
  heatmap:         null,   // base64 PNG string
  network:         null,   // base64 PNG string
  vectors:         null,   // {terms: [], vectors: {doc: [weights]}, total_features, shown_features}
  topTerms:        null,   // {filename: [[term, weight], ...]}
  vectorSpace:     null,   // base64 PNG string (PCA projection)
  error:           null,
};

// ── Reducer ───────────────────────────────────────────────────────────────────
function reducer(state, action) {
  switch (action.type) {
    case "SET_FILES":
      return { ...state, uploadedFiles: action.payload, error: null };
    case "SET_FILE_METADATA":
      return { ...state, fileMetadata: action.payload };
    case "SET_THRESHOLD":
      return { ...state, threshold: action.payload };
    case "SET_UPLOADING":
      return { ...state, isUploading: action.payload };
    case "SET_ANALYZING":
      return { ...state, isAnalyzing: action.payload };
    case "SET_PIPELINE":
      return { ...state, pipelineSteps: action.payload };
    case "STEP_ACTIVE":
      return {
        ...state,
        pipelineSteps: state.pipelineSteps.map((s, i) =>
          i === action.payload ? { ...s, status: "active" } : s
        ),
      };
    case "STEP_DONE":
      return {
        ...state,
        pipelineSteps: state.pipelineSteps.map((s, i) =>
          i <= action.payload ? { ...s, status: "done" } : s
        ),
      };
    case "SET_RESULTS":
      return { ...state, results: action.payload };
    case "SET_EXPLANATIONS":
      return { ...state, explanations: action.payload };
    case "SET_SENTENCES":
      return { ...state, sentences: action.payload };
    case "SET_PROCESSED":
      return { ...state, processed: action.payload };
    case "SET_HEATMAP":
      return { ...state, heatmap: action.payload };
    case "SET_NETWORK":
      return { ...state, network: action.payload };
    case "SET_VECTORS":
      return { ...state, vectors: action.payload };
    case "SET_TOP_TERMS":
      return { ...state, topTerms: action.payload };
    case "SET_VECTOR_SPACE":
      return { ...state, vectorSpace: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload, isAnalyzing: false, isUploading: false };
    case "RESET_ANALYSIS":
      return {
        ...state,
        results: null, explanations: null, sentences: null,
        processed: null, heatmap: null, network: null,
        vectors: null, topTerms: null, vectorSpace: null,
        pipelineSteps: [], error: null,
      };
    default:
      return state;
  }
}

// ── Context ───────────────────────────────────────────────────────────────────
const AnalysisContext = createContext(null);

export function AnalysisProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <AnalysisContext.Provider value={{ state, dispatch }}>
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error("useAnalysis must be used inside AnalysisProvider");
  return ctx;
}
