
import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API =
  (import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1").replace(
    /\/$/,
    ""
  );

type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type Verification = {
  claim: {
    type: string;
    value: string;
  };
  status: string;
  reason: string;
};

type InspectionReport = {
  inspection_id: string;
  status: string;
  defect_type: string;
  confidence: number;
  localization: {
    bbox: BoundingBox | null;
    region: string;
    heatmap_png_base64: string;
    overlay_png_base64: string;
  };
  anomaly_score: number;
  explanation: {
    provider: string;
    description: string;
    severity: string;
    evidence: string;
    claims: {
      type: string;
      value: string;
    }[];
  };
  grounding: {
    score: number;
    supported_claims: number;
    total_claims: number;
    hallucination_risk: string;
    verifications: Verification[];
  };
  uncertainty: {
    model_confidence: number;
    grounding_confidence: number;
    evidence_strength: number;
    level: string;
    note: string;
  };
  created_at: string;
};

type ImageTab = "original" | "overlay" | "heatmap";

function formatLabel(value?: string) {
  if (!value) return "Unknown";
  return value.replaceAll("_", " ");
}

function percentage(value?: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function Meter({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  const safeValue = Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : 0;

  return (
    <div className="meter">
      <div>
        <span>{label}</span>
        <b>{percentage(value)}</b>
      </div>

      <i>
        <em style={{ width: `${safeValue * 100}%` }} />
      </i>
    </div>
  );
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [report, setReport] = useState<InspectionReport | null>(null);
  const [tab, setTab] = useState<ImageTab>("original");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  function selectFile(selectedFile?: File) {
    if (!selectedFile) return;

    setError("");
    setReport(null);
    setTab("original");

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    if (!allowedTypes.includes(selectedFile.type)) {
      setFile(null);
      setPreview("");
      setError("Please select a JPG, PNG, or WEBP image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setFile(null);
      setPreview("");
      setError("Image size must be 10 MB or less.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  }

  async function inspect() {
    if (!file || busy) return;

    setBusy(true);
    setError("");
    setReport(null);

    const body = new FormData();
    body.append("image", file);

    try {
      const response = await fetch(`${API}/inspect`, {
        method: "POST",
        body,
      });

      const data = await response.json();

      if (!response.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : `Inspection failed with status ${response.status}.`;

        throw new Error(message);
      }

      if (!data.defect_type || !data.localization) {
        throw new Error(
          "The server response is missing inspection results."
        );
      }

      setReport(data as InspectionReport);
      setTab("overlay");
    } catch (err) {
      if (err instanceof TypeError) {
        setError(
          "Cannot connect to VisionGuard. Check that the backend is running."
        );
      } else {
        setError(
          err instanceof Error
            ? err.message
            : "An unexpected error occurred."
        );
      }
    } finally {
      setBusy(false);
    }
  }

  const visual =
    tab === "original"
      ? preview
      : tab === "heatmap"
        ? report?.localization.heatmap_png_base64
          ? `data:image/png;base64,${report.localization.heatmap_png_base64}`
          : ""
        : report?.localization.overlay_png_base64
          ? `data:image/png;base64,${report.localization.overlay_png_base64}`
          : "";

  return (
    <main>
      <header>
        <div className="brand">
          VISION<span>GUARD</span>
        </div>

        <p>HALLUCINATION-AWARE VISUAL INSPECTION</p>
        <small>DETECT · LOCALIZE · EXPLAIN · VERIFY</small>
      </header>

      <section className="intro">
        <h1>Evidence-first surface inspection.</h1>
        <p>
          Upload a metal-surface image for defect classification,
          visual localization, and evidence-based explanation.
        </p>
      </section>

      <section className="upload">
        <input
          id="upload"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(event) =>
            selectFile(event.target.files?.[0])
          }
        />

        <label htmlFor="upload">
          {file
            ? file.name
            : "Choose a metal-surface image"}
        </label>

        <button
          disabled={!file || busy}
          onClick={inspect}
        >
          {busy ? "Inspecting…" : "Inspect image"}
        </button>

        {error && (
          <strong className="error" role="alert">
            {error}
          </strong>
        )}
      </section>

      {busy && (
        <p role="status" aria-live="polite">
          Analyzing image. Please wait…
        </p>
      )}

      {report && (
        <>
          <section className="grid">
            <article className="viewer">
              <nav aria-label="Inspection image views">
                {(
                  [
                    ["original", "Original"],
                    ["overlay", "Detection overlay"],
                    ["heatmap", "Grad-CAM heatmap"],
                  ] as [ImageTab, string][]
                ).map(([value, label]) => (
                  <button
                    key={value}
                    className={tab === value ? "active" : ""}
                    onClick={() => setTab(value)}
                    aria-pressed={tab === value}
                  >
                    {label}
                  </button>
                ))}
              </nav>

              {visual ? (
                <img
                  src={visual}
                  alt={`${formatLabel(report.defect_type)} inspection: ${tab}`}
                />
              ) : (
                <p>Visualization unavailable.</p>
              )}

              {report.localization.bbox && (
                <p className="region">
                  Bounding box:{" "}
                  <b>
                    ({report.localization.bbox.x1},{" "}
                    {report.localization.bbox.y1}) – (
                    {report.localization.bbox.x2},{" "}
                    {report.localization.bbox.y2})
                  </b>
                </p>
              )}
            </article>

            <article className="result">
              <div className={`status ${report.status}`}>
                {formatLabel(report.status)}
              </div>

              <h2>{formatLabel(report.defect_type)}</h2>

              <Meter
                label="Model confidence"
                value={report.confidence}
              />

              <Meter
                label="Anomaly response"
                value={report.anomaly_score}
              />

              <p className="region">
                Evidence region:{" "}
                <b>{formatLabel(report.localization.region)}</b>
              </p>

              <p className="region">
                Severity:{" "}
                <b>
                  {formatLabel(report.explanation.severity)}
                </b>
              </p>

              <small>
                Inspection ID: {report.inspection_id}
              </small>
            </article>
          </section>

          <section className="cards">
            <article>
              <h3>AI explanation</h3>

              <p>{report.explanation.description}</p>

              {report.explanation.evidence && (
                <>
                  <h4>Evidence</h4>
                  <p>{report.explanation.evidence}</p>
                </>
              )}

              <small>
                Provider: {report.explanation.provider}
              </small>
            </article>

            <article>
              <h3>
                Grounding verification{" "}
                <b>{percentage(report.grounding.score)}</b>
              </h3>

              <p>
                Supported claims:{" "}
                {report.grounding.supported_claims} /{" "}
                {report.grounding.total_claims}
              </p>

              <p>
                Hallucination risk:{" "}
                <b>
                  {formatLabel(
                    report.grounding.hallucination_risk
                  )}
                </b>
              </p>

              {report.grounding.verifications.map(
                (verification, index) => (
                  <div
                    className={`claim ${verification.status}`}
                    key={`${verification.claim.type}-${index}`}
                  >
                    <b>
                      {verification.status === "supported"
                        ? "✓"
                        : verification.status === "weak"
                          ? "!"
                          : "×"}
                    </b>

                    <span>
                      {verification.claim.type}:{" "}
                      {verification.claim.value}

                      <small>{verification.reason}</small>
                    </span>
                  </div>
                )
              )}
            </article>

            <article>
              <h3>Uncertainty</h3>

              <div
                className={`risk ${report.uncertainty.level}`}
              >
                {formatLabel(report.uncertainty.level)} uncertainty
              </div>

              <Meter
                label="Grounding confidence"
                value={report.uncertainty.grounding_confidence}
              />

              <Meter
                label="Evidence strength"
                value={report.uncertainty.evidence_strength}
              />

              <p>{report.uncertainty.note}</p>
            </article>
          </section>
        </>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);