function AboutSection() {
  return (
    <section className="card about-section" id="about">
      <div className="about-header">
        <p className="eyebrow">About the project</p>
        <h2>End-to-end license plate recognition</h2>

        <p>
          This application detects a license plate in an uploaded
          vehicle image, extracts the plate region, enhances the crop,
          and uses optical character recognition to predict the plate
          text.
        </p>
      </div>

      <div className="about-stat-grid">
        <AboutStat value="4,500" label="Dataset images" />
        <AboutStat value="150" label="Vehicle tracks" />
        <AboutStat value="960 × 960" label="Detector input" />
        <AboutStat value="78.0%" label="Validation mAP@50–95" />
      </div>

      <div className="pipeline-section">
        <h3>How it works</h3>

        <div className="pipeline-grid">
          <PipelineStep
            number="1"
            title="Upload"
            description="The user uploads a vehicle image."
          />

          <PipelineStep
            number="2"
            title="Detect"
            description="YOLO identifies the highest-confidence plate."
          />

          <PipelineStep
            number="3"
            title="Enhance"
            description="The crop is padded, resized, and enhanced with CLAHE."
          />

          <PipelineStep
            number="4"
            title="Recognize"
            description="Tesseract converts the plate crop into text."
          />
        </div>
      </div>

      <div className="about-detail-grid">
        <article>
          <h3>What the analysis showed</h3>

          <p>
            Increasing detector resolution improved strict bounding-box
            localization and reduced missing crops. However, better
            localization did not produce a statistically significant
            improvement in OCR accuracy.
          </p>

          <p>
            The confidence histogram compares each uploaded image
            against successful validation detections. The percentile is
            a detector-confidence rank, not the probability that the
            recognized text is correct.
          </p>
        </article>

        <article>
          <h3>Current limitations</h3>

          <p>
            The system was evaluated primarily using the UFPR-ALPR
            dataset. Results may be weaker for other countries, unusual
            plate formats, nighttime images, blur, distant vehicles, or
            severe viewing angles.
          </p>

          <p>
            The application currently returns only the
            highest-confidence detection. OCR remains the main
            performance bottleneck.
          </p>
        </article>
      </div>

      <div className="technology-section">
        <h3>Technology</h3>

        <div className="technology-list">
          <span>Python</span>
          <span>FastAPI</span>
          <span>YOLO</span>
          <span>PyTorch</span>
          <span>OpenCV</span>
          <span>Tesseract</span>
          <span>React</span>
          <span>Docker</span>
        </div>
      </div>
    </section>
  );
}

function AboutStat({ value, label }) {
  return (
    <div className="about-stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function PipelineStep({ number, title, description }) {
  return (
    <article className="pipeline-step">
      <span className="pipeline-number">{number}</span>
      <h4>{title}</h4>
      <p>{description}</p>
    </article>
  );
}

export default AboutSection;