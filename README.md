# VisionGuard

> Hallucination-aware visual inspection for metal-surface defects.

VisionGuard is an evidence-first inspection application that classifies a metal-surface image, localizes model evidence with Grad-CAM, produces a constrained explanation, and independently verifies the explanation's defect and location claims. It deliberately does **not** ship with model weights or report fabricated metrics: train a checkpoint on your dataset before calling the inspection endpoint.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Design Decisions](#design-decisions)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Training](#training)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [API](#api)
- [Explainability](#explainability)
- [Grounding Verification](#grounding-verification)
- [Model Confidence and Uncertainty](#model-confidence-and-uncertainty)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Docker](#docker)
- [Security and Privacy](#security-and-privacy)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Interview Explanation](#interview-explanation)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Overview

The application follows a **detect → localize → explain → verify** workflow. A locally trained ResNet-18 supplies the visual prediction. Grad-CAM identifies the image region that influenced that prediction. A local, template-based narrator converts only those values into structured claims, and the grounding verifier marks each claim as supported, weak, or unsupported.

The supported classifier categories are `crack`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, and `scratches`. The project is intended as a baseline for inspection workflows—not as a substitute for qualified human review.

## Features

- ResNet-18 classification of seven NEU-style surface categories.
- Grad-CAM heatmap, overlay, and coarse evidence bounding box.
- Softmax confidence and non-normal anomaly response.
- Constrained, local evidence narration rather than free-form model text.
- Semantic defect-claim and spatial location-claim verification.
- Claim-level grounding score and hallucination-risk label.
- SQLite-backed inspection history lookup.
- Validated JPG/JPEG, PNG, and WEBP uploads.
- FastAPI backend, React + TypeScript frontend, automated tests, and Docker support.

## Architecture

```text
Input image
    │
    ▼
Upload validation ──► RGB conversion and ImageNet normalization
    │
    ▼
ResNet-18 classifier ──► predicted class + softmax confidence
    │
    ├──────────────────► Grad-CAM heatmap/overlay ──► evidence bounding box
    │
    ▼
Local evidence narrator ──► structured defect, location, and severity claims
    │
    ▼
Grounding verifier ──► semantic + spatial checks ──► score and risk label
    │
    ▼
FastAPI response, SQLite history, and React UI
```

## How It Works

1. The backend rejects empty, corrupt, oversized, mismatched-extension, or unsupported uploads.
2. The image is converted to RGB, resized to `224 × 224`, converted to a tensor, and normalized with ImageNet statistics.
3. The checkpointed ResNet-18 predicts one of the seven categories using softmax probabilities.
4. Grad-CAM from the final ResNet block is normalized, rendered as a heatmap and overlay, and thresholded for a coarse evidence box for non-`normal` results.
5. The local narrator makes only claims based on the prediction, evidence region, confidence, and anomaly response.
6. The grounding verifier compares defect words with a controlled synonym taxonomy and verifies location using whether the evidence-box center is inside the named region.
7. The API persists the completed response as inspection history and returns it to the UI.

## Design Decisions

- **Evidence before language:** classification and localization are completed before an explanation is created.
- **Constrained narrator:** `local-evidence-narrator` is a deterministic local formatter, not a VLM. A future text-generation adapter must still return the same structured claim schema.
- **Verification is independent:** narrative claims are checked against classifier output and Grad-CAM evidence rather than trusted by default.
- **Center-based location checks:** a small defect can be correctly located in a quadrant even if its bounding-box IoU with the full quadrant is low. IoU is returned as diagnostics, while the evidence-box center determines support.
- **No invented severity:** severity is always marked `weak` because this baseline has no trained severity target; it does not affect the grounding score.
- **Fail closed:** inspection requests receive `503` when a trained checkpoint is unavailable rather than synthetic predictions.

## Technology Stack

| Area | Technology |
| --- | --- |
| API | FastAPI, Uvicorn, Pydantic |
| ML | PyTorch, Torchvision ResNet-18, scikit-learn |
| Image processing | Pillow, OpenCV, NumPy |
| Explainability | Gradient-weighted Class Activation Mapping (Grad-CAM) |
| Persistence | SQLite via SQLAlchemy |
| Frontend | React, TypeScript, Vite |
| Deployment | Docker and Docker Compose |
| Tests | pytest, Vitest |

## Project Structure

```text
VisionGuard/
├── backend/
│   ├── app/
│   │   ├── grounding/       # Claim taxonomy and verification
│   │   ├── ml/              # Model, preprocessing, and Grad-CAM
│   │   ├── schemas/         # API response models
│   │   ├── services/        # Inspection and explanation orchestration
│   │   ├── config.py        # Environment-backed settings
│   │   ├── database.py      # SQLite inspection history
│   │   └── main.py          # FastAPI routes
│   ├── tests/
│   └── requirements.txt
├── frontend/                # Vite React application
├── ml/training/train.py     # Training entry point
├── docs/                    # Model and grounding notes
└── docker-compose.yml
```

## Dataset

Training expects an ImageFolder-compatible dataset with the NEU Surface Defect Database category names plus a `normal` class. Dataset files are not included in this repository and `ml/data/` is ignored by Git. You are responsible for obtaining the dataset and ensuring that you have the right to use it.

The trainer requires the exact class set below. Directory names are sorted by Torchvision's `ImageFolder`, so the names must match exactly to preserve the model's class-index order.

## Dataset Structure

```text
ml/data/neu/
├── train/
│   ├── crack/
│   ├── inclusion/
│   ├── patches/
│   ├── pitted_surface/
│   ├── rolled-in_scale/
│   └── scratches/
├── val/
│   └── <the same seven class directories>/
└── test/
    └── <the same seven class directories>/
```

The current training script consumes `train/` and `val/`. Keep `test/` separate for final evaluation.

## Installation

**Prerequisites:** Python 3.11+ (the backend container uses 3.11), Node.js 22+ for the supplied frontend Docker build, and optionally Docker Compose.

```bash
git clone <your-fork-or-repository-url>
cd VisionGuard

python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

cd frontend
npm install
cd ..
```

Optional backend settings can be supplied in a `.env` file in the backend process working directory or as environment variables:

```dotenv
MODEL_PATH=../ml/saved_models/visionguard_resnet18.pt
DATABASE_URL=sqlite:///./visionguard.db
ALLOWED_ORIGINS=http://localhost:5173
MAX_UPLOAD_BYTES=10485760
MAX_IMAGE_PIXELS=4000000
MAX_IMAGE_DIMENSION=4096
```

`DATABASE_URL` is intentionally restricted to local `sqlite:///` URLs.

## Training

Place the dataset in the structure above, then run the trainer from the repository root:

```bash
python ml/training/train.py \
  --dataset ml/data/neu \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --output ml/saved_models/visionguard_resnet18.pt
```

The trainer initializes ImageNet-pretrained ResNet-18, uses cross-entropy loss and AdamW, applies random horizontal flips only during training, and retains the checkpoint with the best validation weighted F1. It writes validation metrics to `ml/saved_models/visionguard_resnet18.metrics.json`. Checkpoints and datasets are ignored by Git.

## Running the Backend

Run from `backend/` so the default relative checkpoint and SQLite paths resolve as documented:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Confirm service and checkpoint readiness:

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/model-info
```

`/api/v1/health` returns `model_ready: false` until the checkpoint path exists. Train or configure `MODEL_PATH` before submitting inspections.

## Running the Frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Open the URL reported by Vite (normally `http://localhost:5173`). By default, the frontend calls `http://localhost:8000/api/v1`. Override that API prefix when starting Vite if needed:

```bash
VITE_API_URL=http://localhost:8000/api/v1 npm run dev
```

## API

Base URL: `http://localhost:8000/api/v1`

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status and checkpoint availability. |
| `GET` | `/model-info` | Model name, categories, stored validation metrics, and checkpoint state. |
| `POST` | `/inspect` | Inspect a multipart `image` upload. |
| `GET` | `/inspection/{inspection_id}` | Retrieve a persisted inspection response. |

Example inspection request:

```bash
curl -X POST http://localhost:8000/api/v1/inspect \
  -F 'image=@surface.png'
```

A successful inspection contains an ID, `normal` or `defective` status, predicted `defect_type`, `confidence`, `anomaly_score`, localization images encoded as base64 PNG, explanation claims, grounding verifications, uncertainty, and a UTC creation timestamp. Invalid images return `422`; a missing checkpoint returns `503`; unknown history IDs return `404`.

## Explainability

VisionGuard calculates Grad-CAM from `model.layer4[-1].conv2`, the final ResNet convolutional block, for the predicted class. The normalized map is rendered both as a heatmap and blended overlay. A threshold of `max(0.35, CAM mean + CAM standard deviation)` produces the coarse evidence box for non-normal predictions.

Grad-CAM indicates which image areas influenced the classifier; it is not segmentation and must not be interpreted as a pixel-accurate defect mask. See [the model notes](docs/model.md) for evaluation guidance.

## Grounding Verification

Each explanation is represented as typed claims:

- **Defect claims** are normalized through a controlled synonym map—for example, `fracture` supports a `crack` prediction.
- **Location claims** are checked against one of five named regions: `upper-left`, `upper-right`, `center`, `bottom-left`, and `bottom-right`.
- **Severity claims** are reported as weak because severity is not a classifier output.

The grounding score is the mean evidence confidence across verifiable defect and location claims. Risk is `low` for a fully supported score of at least 0.75, `medium` at 0.4 or above, and `high` otherwise. Additional methodology is in [the grounding notes](docs/grounding.md).

## Model Confidence and Uncertainty

`confidence` is the predicted class's softmax score. It is a model output, **not a calibrated probability that the classification is correct**. `anomaly_score` is `1 - P(normal)`, so it measures the classifier's total probability of non-normal classes and is not derived from the Grad-CAM image.

The returned uncertainty level combines model confidence and grounding score:

| Level | Rule |
| --- | --- |
| `low` | confidence ≥ 0.80 and grounding score ≥ 0.70 |
| `medium` | otherwise, confidence ≥ 0.55 |
| `high` | confidence < 0.55 |

Use these signals to prioritize human review, especially for medium or high uncertainty and unsupported claims.

## Evaluation

The trainer reports validation accuracy plus weighted precision, recall, and F1. For a credible final report, evaluate the selected checkpoint once on the held-out `test/` split using the same preprocessing and report the actual metrics, class distribution, confusion matrix, and evaluation protocol.

If annotated defect masks are available, compare the Grad-CAM-derived evidence box with masks using IoU as a diagnostic localization measure. Do not present Grad-CAM as a replacement for a trained segmentation model or claim performance values that have not been measured on your data.

## Testing

Run backend tests from `backend/`:

```bash
cd backend
pytest -q
```

They cover image loading and limits, filename/magic-byte agreement, IoU behavior, semantic verification, location support, and empty-claim handling.

Run the frontend production build from `frontend/`:

```bash
cd frontend
npm run build
```

## Docker

After training, ensure the checkpoint is present at `ml/saved_models/visionguard_resnet18.pt`, then start both services from the repository root:

```bash
docker compose up --build
```

The backend is exposed at `http://localhost:8000`; the frontend is exposed at `http://localhost:5173`. Compose mounts `./ml/saved_models` into the backend at `/ml/saved_models` and sets `MODEL_PATH` accordingly. Stop services with:

```bash
docker compose down
```

## Security and Privacy

- Only JPEG, PNG, and WEBP images are accepted; decoded format is checked against the filename extension when one is supplied.
- Uploads are limited to 10 MB by default, at least `16 × 16` pixels, a maximum 4,096 pixels per side, and four million total pixels.
- The service uses Pillow verification and catches corrupt/decompression-bomb image errors before inference.
- Checkpoints are loaded with PyTorch `weights_only=True` and must contain a tensor state dictionary.
- CORS allows only the configured origin by default (`http://localhost:5173`), and database URLs are restricted to local SQLite.
- Inspection responses, including filename, predictions, explanation, and base64 evidence images, are stored in the configured SQLite database. Treat that database as potentially sensitive and secure or delete it according to your retention policy.

## Limitations

- Model quality depends on the representativeness, labels, and split quality of the dataset you train on.
- The supplied baseline is a classifier; its localization is coarse Grad-CAM evidence, not segmentation.
- Softmax scores are not calibrated probabilities of correctness.
- Severity has no dedicated trained output.
- The UI has no authentication, authorization, rate limiting, or production-grade observability.
- SQLite and the default configuration target local development, not a multi-user production deployment.
- The application has no included checkpoint or guaranteed performance metrics.

## Roadmap

- Add repeatable test-set evaluation, confusion-matrix export, and metric reporting.
- Support calibrated confidence and abstention thresholds.
- Add segmentation or detection models with mask/box supervision.
- Add a structured VLM adapter while retaining independent claim verification.
- Add authenticated multi-user deployments, retention controls, and audit logging.
- Expand frontend accessibility, history browsing, and export workflows.

## Interview Explanation

> VisionGuard separates seeing from saying. A ResNet-18 first classifies the metal surface, and Grad-CAM exposes the region that drove that classification. A constrained narrator turns those concrete outputs into typed claims instead of generating unconstrained prose. The verifier then independently checks the defect term against a synonym taxonomy and the location against the evidence box. This makes an explanation inspectable: unsupported claims are visible rather than silently presented as facts. The design intentionally treats confidence as uncalibrated and Grad-CAM as coarse evidence, so human review remains part of the decision process.

## Contributing

1. Fork the repository and create a focused branch.
2. Keep changes consistent with the evidence-first design and avoid unmeasured performance claims.
3. Add or update tests for behavior changes.
4. Run the backend tests and frontend build before opening a pull request.
5. Describe dataset assumptions, model changes, and validation evidence in the pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

DAMACHARLA SUSHMA.
