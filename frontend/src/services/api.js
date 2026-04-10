import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 120000,
});

/** Upload multiple TXT files (FormData with field 'files') */
export const uploadFiles = (files) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return api.post("/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

/** Run the full NLP analysis pipeline */
export const analyzeDocuments = (threshold = 0.75) =>
  api.post(`/analyze?threshold=${threshold}`);

/** Fetch similarity matrix + suspicious pairs */
export const getResults = () => api.get("/results");

/** Fetch heatmap as base64 PNG */
export const getHeatmap = () => api.get("/heatmap");

/** Fetch network graph as base64 PNG */
export const getNetwork = () => api.get("/network");

/** Fetch sentence-level matches */
export const getSentences = () => api.get("/sentences");

/** Fetch preprocessed token preview per document */
export const getProcessed = () => api.get("/processed");

/** Fetch shared n-gram explanations per suspicious pair */
export const getExplanations = () => api.get("/explanations");

/** Fetch TF-IDF feature matrix (top-N terms × documents) */
export const getVectors = (topN = 40) => api.get(`/vectors?top_n=${topN}`);

/** Fetch top-10 TF-IDF weighted terms per document */
export const getTopTerms = () => api.get("/top-terms");

/** Fetch PCA 2D projection of TF-IDF vectors as base64 PNG */
export const getVectorSpace = () => api.get("/vector-space");

/** Fetch clustered network graph with community detection as base64 PNG */
export const getClusteredNetwork = (threshold = 0.75, hideSingletons = false) =>
  api.get(`/network-clustered?threshold=${threshold}&hide_singletons=${hideSingletons}`);

/** Health check */
export const healthCheck = () => api.get("/health");
