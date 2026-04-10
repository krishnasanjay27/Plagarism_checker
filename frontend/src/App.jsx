import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AnalysisProvider } from "./context/AnalysisContext";
import Sidebar from "./components/Sidebar";
import UploadPage from "./pages/UploadPage";
import ResultsPage from "./pages/ResultsPage";
import VisualizationPage from "./pages/VisualizationPage";

export default function App() {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <div className="app-shell">
          <Sidebar />
          <main className="app-main">
            <Routes>
              <Route path="/"               element={<UploadPage />} />
              <Route path="/results"        element={<ResultsPage />} />
              <Route path="/visualizations" element={<VisualizationPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </AnalysisProvider>
  );
}
