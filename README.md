# VisionGuard

**Hallucination-aware visual inspection for metal-surface defects.** VisionGuard is a reproducible portfolio system that deliberately separates visual inference from language: **detect → localize → explain → verify**. Its classifier is a locally trained PyTorch model; it never uploads inspection images to a third-party AI service.

> This repository contains code, not fabricated experiment results or model weights. Train against the documented dataset before using `/inspect`.

## Architecture
```text
image → validation/RGB/normalization → ResNet-18 classifier → Grad-CAM evidence map
                                        ↓                         ↓
                              defect + softmax confidence      bounding box
                                        ↓                         ↓
                      local evidence narrator (replaceable VLM interface)
                                        ↓
                      structured claims → semantic + spatial verifier
                                        ↓
                    grounding score / unsupported claims / uncertainty → UI
```

## Design decisions
- **Domain/dataset:** the initial taxonomy follows the [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html): normal plus crack, inclusion, patches, pitted surface, rolled-in scale, and scratches. Obtain the dataset from its official source and review its terms before use.
- **Model:** ImageNet-initialized ResNet-18 is small enough for a developer machine while retaining convolutional features required for Grad-CAM. Classification confidence is the actual softmax output from the trained checkpoint.
- **Localization:** Grad-CAM from `layer4` is thresholded into an evidence box and a heatmap. It is attribution, not a segmentation model; use pixel masks and a segmentation model when production localization accuracy is required. The anomaly score is the classifier's non-`normal` probability—not the normalized heatmap maximum.
- **Grounding:** defect words are resolved through a synonym taxonomy (replaceable with embeddings); named regions become image rectangles and are checked against the evidence box by IoU. Score = mean claim evidence confidence. Severity is intentionally *weak* because the baseline has no trained severity target.

## Project layout
```text
backend/app/{ml,grounding,services}  # inference and independent evidence verifier
backend/tests                       # unit tests
ml/training/train.py                # reproducible classifier training
frontend/src                        # React inspection dashboard
docs/                               # architecture and evaluation notes
```

## Setup and training
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
```
Arrange validated files as `DATASET/{train,val,test}/{normal,crack,inclusion,patches,pitted_surface,rolled-in_scale,scratches}`. Use a stratified split script appropriate to your source provenance; never mix related source images between splits.

```bash
PYTHONPATH=. python ml/training/train.py --dataset /path/to/DATASET --epochs 15 --batch-size 32 --learning-rate 0.001 --output ml/saved_models/visionguard_resnet18.pt
```
The command writes weights, train configuration, and measured validation metrics. Evaluate a held-out test split separately; do not describe validation metrics as test metrics.

## Run
```bash
cd backend && MODEL_PATH=../ml/saved_models/visionguard_resnet18.pt uvicorn app.main:app --reload
cd frontend && npm install && npm run dev
pytest backend/tests
docker compose up --build
```

`POST /api/v1/inspect` accepts a JPG, PNG, or WEBP multipart field named `image` (10 MB and 4 million decoded pixels maximum). Its filename extension must agree with the image's decoded format. `GET /api/v1/health`, `GET /api/v1/model-info`, and `GET /api/v1/inspection/{id}` support operations and history.

```bash
curl -F image=@surface.png http://localhost:8000/api/v1/inspect
```

## Evaluation
Report only values calculated by `train.py` or dedicated held-out evaluation: accuracy, precision, recall, F1 and confusion matrix for classification; IoU of attribution boxes against available masks for localization; image/pixel ROC-AUC where mask scores are implemented. Grounding evaluation requires labeled claims and should report claim support accuracy, spatial accuracy, and unsupported-claim precision/recall. No experimental values are claimed in this repository.

## Limitations and roadmap
Grad-CAM is not a defect segmenter, softmax is not calibrated uncertainty, and the local narrator is evidence-constrained rather than a general VLM. Add a configurable VLM adapter only with explicit image-sharing consent, calibrated probabilities, MVTec mask evaluation, human review queues, and additional industrial domains.

## Interview summary
VisionGuard demonstrates a practical safeguard for multimodal systems: independent computer-vision evidence is converted into structured semantic/spatial tests before a language explanation is trusted. Unsupported language is flagged as *potentially unsupported*, never claimed as perfectly detected hallucination.
