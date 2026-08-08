# License Plate Recognition Web Application

An end-to-end computer vision application for detecting and recognizing vehicle license plates from uploaded images.

The project combines a YOLO-based plate detector, image preprocessing, Tesseract OCR, a FastAPI backend, a React frontend, and Docker-based deployment. In addition to returning a plate prediction, the web interface exposes model confidence, validation percentile, bounding-box statistics, and inference latency.

![Application demo](demo.png)

## Features

- Upload vehicle images through a React interface
- Detect the highest-confidence license plate with YOLO
- Draw the predicted bounding box on the uploaded image
- Crop and preprocess the detected plate region
- Recognize plate text with Tesseract OCR
- Display detector confidence and validation percentile
- Compare each prediction with the validation confidence distribution
- Show detection, OCR, and total inference latency
- Expose prediction and model metadata through FastAPI
- Run the full application with Docker Compose

The frontend is served by Nginx in production. Requests to `/api/*` are proxied internally to the FastAPI backend.

## Dataset

The project was developed using the **UFPR-ALPR** dataset.

| Statistic | Value |
|---|---:|
| Images | 4,500 |
| Vehicle tracks | 150 |
| Detector input size | 960 × 960 |
| Validation images used for analysis | 900 |

Vehicle tracks were kept grouped during evaluation so related frames from the same vehicle were not treated as independent examples across splits.

## Model Pipeline

### 1. License Plate Detection

The deployed detector uses the 960-pixel YOLO experiment with:

- Input size: `960`
- Confidence threshold: `0.25`
- Maximum detections considered: `5`
- Final prediction: highest-confidence bounding box

The 960 model was selected because it improved strict localization quality and reduced missing plate crops.

### 2. Plate Preprocessing

The detected crop is processed with:

- 5% bounding-box padding
- Grayscale conversion
- Resize to 96-pixel height
- Bicubic interpolation
- CLAHE contrast enhancement
- 12-pixel white border

### 3. OCR

Tesseract uses:

```text
--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
```

The OCR configuration is:

```text
tesseract_psm7_clahe_h96_pad005_v1
```

## Validation Results

### Detector Performance

| Metric | 640 Model | 960 Model |
|---|---:|---:|
| Precision | 96.06% | 94.29% |
| Recall | 97.47% | 96.78% |
| mAP@0.50 | 99.22% | 98.96% |
| mAP@0.50:0.95 | 73.74% | **78.04%** |

Although the 640 model had slightly stronger precision, recall, and mAP@0.50, the 960 model achieved better strict localization performance.

### OCR Evaluation

| Crop Source | Crop Availability | Exact Match | Normalized Similarity |
|---|---:|---:|---:|
| Ground-truth crop | 100.00% | 8.56% | 24.32% |
| YOLO 640 crop | 97.11% | 11.11% | 24.66% |
| YOLO 960 crop | **99.78%** | 8.22% | **25.02%** |

Increasing detector resolution improved localization and crop availability, but it did not produce a statistically significant improvement in OCR performance.

At the vehicle-track level, the Spearman correlation between improvement in localization IoU and improvement in OCR similarity was approximately:

```text
rho = 0.0009
p = 0.996
```

This suggests that the current performance bottleneck is primarily the recognition stage rather than plate localization.

## Technology Stack

### Machine Learning / Computer Vision

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- Pandas
- Tesseract OCR
- pytesseract

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- React
- Vite
- JavaScript
- CSS
- Nginx

### Deployment

- Docker
- Docker Compose

## Run with Docker

### Requirements

Install:

- Docker Desktop
- Docker Compose

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd Plate-Recognition
```

Build and start the application:

```bash
docker compose up -d --build
```

Check the containers:

```bash
docker compose ps
```

Open the web application:

```text
http://localhost:3000
```

The backend API documentation is available locally at:

```text
http://localhost:8000/docs
```

Follow logs with:

```bash
docker compose logs -f
```

Stop the application with:

```bash
docker compose down
```

## Local Development

### Backend

Create and activate a Python virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Frontend

From the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## Limitations

- The system was evaluated primarily on UFPR-ALPR images.
- Performance may decrease for plate styles that differ significantly from the training data.
- Nighttime images, blur, occlusion, distant vehicles, and severe viewing angles may reduce accuracy.
- The current application returns only the highest-confidence plate detection.
- Detector confidence is not a measure of OCR correctness.
- Tesseract exact-match accuracy is currently limited and remains the main system bottleneck.
- CPU inference is slower than GPU inference.

## Future Work

Potential improvements include:

- Replace Tesseract with a stronger learned OCR model
- Support multiple license plates in a single image
- Add perspective correction before OCR
- Improve robustness across countries and plate formats
- Benchmark median and p95 production latency
- Add continuous integration for automated testing
- Deploy the Docker application to a public cloud environment

## License

This repository contains project source code and trained model weights. The UFPR-ALPR dataset is not included. Refer to the original dataset terms for dataset usage and licensing.
