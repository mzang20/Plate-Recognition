import { useEffect, useState } from "react";

import {
  getReferenceStats,
  predictPlate,
} from "./services/api";

import AboutSection from "./components/AboutSection";

import "./App.css";


function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [referenceStats, setReferenceStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getReferenceStats()
      .then(setReferenceStats)
      .catch(() => setReferenceStats(null));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setFile(selectedFile);
    setPreviewUrl(
      URL.createObjectURL(selectedFile)
    );
    setPrediction(null);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      setError("Select an image first.");
      return;
    }

    setLoading(true);
    setPrediction(null);
    setError("");

    try {
      const result = await predictPlate(file);
      setPrediction(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const boundingBoxStyle =
    prediction?.plate_detected &&
    prediction?.bbox &&
    prediction.image_width &&
    prediction.image_height
      ? {
          left: `${
            prediction.bbox.xmin /
            prediction.image_width *
            100
          }%`,
          top: `${
            prediction.bbox.ymin /
            prediction.image_height *
            100
          }%`,
          width: `${
            (
              prediction.bbox.xmax -
              prediction.bbox.xmin
            ) /
            prediction.image_width *
            100
          }%`,
          height: `${
            (
              prediction.bbox.ymax -
              prediction.bbox.ymin
            ) /
            prediction.image_height *
            100
          }%`,
        }
      : null;

  const boxMetrics = calculateBoxMetrics(
    prediction
  );

  return (
    <main className="page">
      <section className="card" id="predict">
        <nav className="site-nav" aria-label="Page navigation">
          <a href="#predict">Predict</a>
          <a href="#about">About</a>
        </nav>
        <div className="heading">
          <p className="eyebrow">
            Computer Vision Demo
          </p>

          <h1>License Plate Recognition</h1>

          <p className="subtitle">
            Upload a vehicle image to detect and
            recognize its license plate.
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="upload-form"
        >
          <label className="upload-box">
            <span>
              {file
                ? file.name
                : "Choose a vehicle image"}
            </span>

            <input
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp"
              onChange={handleFileChange}
            />
          </label>

          <button
            type="submit"
            disabled={!file || loading}
          >
            {loading
              ? "Recognizing..."
              : "Recognize Plate"}
          </button>
        </form>

        {error && (
          <p className="error-message">
            {error}
          </p>
        )}

        {previewUrl && (
          <div className="image-section">
            <div className="image-wrapper">
              <img
                src={previewUrl}
                alt="Uploaded vehicle"
              />

              {boundingBoxStyle && (
                <div
                  className="bounding-box"
                  style={boundingBoxStyle}
                >
                  <span>
                    {prediction.plate_text ||
                      "Plate"}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {prediction && (
          <section className="results">
            <h2>Prediction</h2>

            {prediction.plate_detected ? (
              <>
                <div className="plate-result">
                  {prediction.plate_text ||
                    "Text not recognized"}
                </div>

                <div className="metrics-grid">
                  <Metric
                    label="Detector confidence"
                    value={formatPercentage(
                      prediction.detector_confidence
                    )}
                  />

                  <Metric
                    label="Validation percentile"
                    value={
                      prediction.confidence_percentile !==
                      null
                        ? `${prediction.confidence_percentile.toFixed(
                            1
                          )}%`
                        : "N/A"
                    }
                  />

                  <Metric
                    label="Total latency"
                    value={`${prediction.latency.total_ms.toFixed(
                      1
                    )} ms`}
                  />

                  <Metric
                    label="Detected box"
                    value={`${boxMetrics.width} × ${boxMetrics.height} px`}
                  />

                  <Metric
                    label="Box aspect ratio"
                    value={boxMetrics.aspectRatio.toFixed(
                      2
                    )}
                  />

                  <Metric
                    label="Image area occupied"
                    value={`${boxMetrics.areaRatio.toFixed(
                      3
                    )}%`}
                  />
                </div>

                <div className="analytics-grid">
                  <ConfidenceDistribution
                    distribution={
                      referenceStats?.detector_confidence
                    }
                    confidence={
                      prediction.detector_confidence
                    }
                    percentile={
                      prediction.confidence_percentile
                    }
                  />

                  <LatencyBreakdown
                    latency={prediction.latency}
                  />
                </div>
              </>
            ) : (
              <p className="no-detection">
                No license plate was detected.
              </p>
            )}
          </section>
        )}
      </section>
      <AboutSection />
    </main>
  );
}


function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}


function ConfidenceDistribution({
  distribution,
  confidence,
  percentile,
}) {
  if (!distribution || confidence == null) {
    return null;
  }

  const maximumCount = Math.max(
    ...distribution.bin_counts,
    1
  );

  const markerPosition = Math.min(
    99.5,
    Math.max(0.5, confidence * 100)
  );

  return (
    <section className="analytics-card">
      <div className="analytics-heading">
        <div>
          <h3>Confidence distribution</h3>
          <p>
            Compared with successful validation
            detections
          </p>
        </div>

        <strong>
          {(confidence * 100).toFixed(1)}%
        </strong>
      </div>

      <div className="histogram">
        {distribution.bin_counts.map(
          (count, index) => {
            const lower =
              distribution.bin_edges[index];

            const upper =
              distribution.bin_edges[index + 1];

            return (
              <div
                className="histogram-column"
                key={`${lower}-${upper}`}
                title={`${(
                  lower * 100
                ).toFixed(0)}–${(
                  upper * 100
                ).toFixed(0)}%: ${count} images`}
              >
                <div
                  className="histogram-bar"
                  style={{
                    height: `${Math.max(
                      3,
                      count /
                        maximumCount *
                        100
                    )}%`,
                  }}
                />
              </div>
            );
          }
        )}

        <div
          className="histogram-marker"
          style={{
            left: `${markerPosition}%`,
          }}
        >
          <span>Your image</span>
        </div>
      </div>

      <div className="histogram-axis">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>

      <p className="analytics-note">
        Higher than{" "}
        <strong>
          {percentile?.toFixed(1)}%
        </strong>{" "}
        of {distribution.sample_size} successful
        validation detections.
      </p>
    </section>
  );
}


function LatencyBreakdown({ latency }) {
  const total = Math.max(
    latency.total_ms,
    0.001
  );

  const detectionPercent =
    latency.detection_ms / total * 100;

  const ocrPercent =
    latency.ocr_ms / total * 100;

  const overheadMs = Math.max(
    0,
    latency.total_ms -
      latency.detection_ms -
      latency.ocr_ms
  );

  const overheadPercent =
    overheadMs / total * 100;

  return (
    <section className="analytics-card">
      <div className="analytics-heading">
        <div>
          <h3>Latency breakdown</h3>
          <p>Processing time by pipeline stage</p>
        </div>

        <strong>
          {latency.total_ms.toFixed(1)} ms
        </strong>
      </div>

      <div className="latency-bar">
        <div
          className="latency-segment detection-segment"
          style={{
            width: `${detectionPercent}%`,
          }}
        />

        <div
          className="latency-segment ocr-segment"
          style={{
            width: `${ocrPercent}%`,
          }}
        />

        <div
          className="latency-segment overhead-segment"
          style={{
            width: `${overheadPercent}%`,
          }}
        />
      </div>

      <div className="latency-list">
        <LatencyItem
          label="Detection"
          value={latency.detection_ms}
          className="detection-dot"
        />

        <LatencyItem
          label="OCR"
          value={latency.ocr_ms}
          className="ocr-dot"
        />

        <LatencyItem
          label="Other"
          value={overheadMs}
          className="overhead-dot"
        />
      </div>
    </section>
  );
}


function LatencyItem({
  label,
  value,
  className,
}) {
  return (
    <div className="latency-item">
      <span className={`latency-dot ${className}`} />

      <span>{label}</span>

      <strong>{value.toFixed(1)} ms</strong>
    </div>
  );
}


function calculateBoxMetrics(prediction) {
  if (
    !prediction?.bbox ||
    !prediction.image_width ||
    !prediction.image_height
  ) {
    return {
      width: 0,
      height: 0,
      aspectRatio: 0,
      areaRatio: 0,
    };
  }

  const width =
    prediction.bbox.xmax -
    prediction.bbox.xmin;

  const height =
    prediction.bbox.ymax -
    prediction.bbox.ymin;

  const aspectRatio =
    height > 0 ? width / height : 0;

  const areaRatio =
    width *
    height /
    (
      prediction.image_width *
      prediction.image_height
    ) *
    100;

  return {
    width,
    height,
    aspectRatio,
    areaRatio,
  };
}


function formatPercentage(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${(value * 100).toFixed(1)}%`;
}

export default App;

