# VisionGuard

### Hallucination-Aware Visual Inspection for Metal-Surface Defects

VisionGuard is a computer-vision-based visual inspection system designed to detect and localize defects on metal surfaces while providing evidence-grounded explanations.

The system follows an evidence-first pipeline:

**Detect → Localize → Explain → Verify**

VisionGuard separates visual inference from language generation. A locally trained PyTorch classifier produces the visual evidence, Grad-CAM provides an evidence map, and an independent grounding layer verifies whether generated claims are supported by the visual evidence.

> **Important:** This repository contains the application and training pipeline. It does not claim fabricated experimental results or include model weights by default. Train the model using the documented dataset before using the inspection endpoint.

---

## ✨ Features

- 🔍 Metal-surface defect classification
- 🧠 ResNet-18 based image classifier
- 🔥 Grad-CAM visual evidence generation
- 📍 Defect localization using evidence regions
- 📊 Softmax confidence reporting
- 💬 Evidence-grounded explanation pipeline
- ✅ Semantic claim verification
- 📐 Spatial claim verification using IoU
- ⚠️ Unsupported-claim detection
- 🛡️ Image upload validation
- 🗂️ Inspection history
- ⚡ FastAPI backend
- ⚛️ React + TypeScript frontend
- 🐳 Docker-based deployment
- 🧪 Automated testing

---

## 🏗️ Architecture

```text
                         Input Image
                              │
                              ▼
                    Image Validation
                              │
                              ▼
                   RGB + Normalization
                              │
                              ▼
                       ResNet-18
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
             Defect Prediction       Grad-CAM
                    │                   │
                    ▼                   ▼
            Softmax Confidence     Evidence Heatmap
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Evidence Region
                              │
                              ▼
                  Local Evidence Narrator
                              │
                              ▼
                     Structured Claims
                              │
                              ▼
                  Grounding Verification
                     ┌────────┴────────┐
                     ▼                 ▼
                Semantic           Spatial
                Verification      Verification
                     │                 │
                     └────────┬────────┘
                              ▼
                     Grounding Score
                              │
                              ▼
                         Frontend UI
