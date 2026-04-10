import React from "react";

const STEPS = [
  "Uploading files",
  "Preprocessing text",
  "Generating TF-IDF vectors",
  "Computing cosine similarity",
  "Rendering visualizations",
];

const StepIcon = ({ status }) => {
  if (status === "done")   return <span>✓</span>;
  if (status === "active") return <span className="spinner" />;
  return <span style={{ color: "var(--text-muted)", fontWeight: 300 }}>○</span>;
};

export default function PipelineStatus({ steps = [] }) {
  if (!steps.length) return null;

  return (
    <div className="pipeline-steps">
      {steps.map((step, i) => (
        <div
          key={i}
          className={
            "pipeline-step" +
            (step.status === "done"   ? " done"   : "") +
            (step.status === "active" ? " active" : "")
          }
        >
          <span className="step-icon">
            <StepIcon status={step.status} />
          </span>
          {step.label}
        </div>
      ))}
    </div>
  );
}

export { STEPS };
