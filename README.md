# MicroAI
An offline, edge-optimized Flutter application for the rapid histopathological triage of colon adenocarcinoma. It executes highly compressed TFLite models directly on smartphones for sub-second, zero-cloud inference.
# MicroAI
**Offline Edge Histopathology Triage for Colon Adenocarcinoma Classification**

A Flutter mobile application that classifies colon tissue histology slides as malignant or benign using an on-device TensorFlow Lite model — no internet connection required.

## Visual Demo

| Input Screen | Malignant Result | OOD Rejection |
| :---: | :---: | :---: |
| <img width="250" alt="Input" src="https://github.com/user-attachments/assets/de8eea34-5bf7-4806-b59b-dbc5d98196fa" /> | <img width="250" alt="Malignant" src="https://github.com/user-attachments/assets/b9425689-cf06-4be0-be82-36f9d586c6fc" /> | <img width="250" alt="Rejection" src="https://github.com/user-attachments/assets/7228f136-bcc9-4125-9972-25470dbdbc594" /> |




## The Problem & Solution
Access to pathologists is severely limited in low-resource clinical settings across Pakistan and the broader developing world. A single histopathology report can take days to weeks when diagnostic infrastructure is unavailable, causing critical delays in cancer triage. Early detection of colon adenocarcinoma, one of the most treatable cancers when caught early, is directly tied to how quickly a slide can be reviewed.

MicroAI addresses this bottleneck by deploying a highly compressed MobileNetV2 model directly onto a standard Android smartphone. A healthcare worker or junior clinician can capture or upload a colon histology slide image and receive a triage-level classification result in under 250 milliseconds, entirely offline, with no cloud dependency. This is not a replacement for a pathologist, it is a first-pass screening tool that flags high-risk cases for priority review.

## Key Features
*   **Zero Cloud Dependency** — 100% offline inference using TensorFlow Lite. The app functions with no internet connection, making it viable in remote or resource-limited environments.
*   **Edge-Optimized Performance** — Sub-250ms classification on standard Android hardware using INT8 quantization, reducing model size without significant accuracy loss.
*   **Two-Stage OOD Detection** — Stage 1 performs H&E color profiling on the raw image pixels to reject non-histology inputs (photos, objects, clocks) before the model even runs. Stage 2 uses post-inference confidence thresholds to detect non-colon tissue slides.
*   **Patient Management** — Full patient profile system with name, age, and analyst identity stored per result. Complete analysis history with timestamps.
*   **Confidence-Aware Results** — Results are tiered into High (≥75%), Moderate (60–75%), and Rejected (<60%) confidence bands with appropriate clinical guidance for each tier.
*   **Research Safeguards** — Every result screen carries a mandatory disclaimer. The app is explicitly scoped to research use only.

## Model & Pipeline Architecture

### Model
| Property | Value |
| :--- | :--- |
| **Base Architecture** | MobileNetV2 |
| **Quantization** | INT8 |
| **Model File Size** | ~4.2 MB |
| **Input Shape** | [1, 224, 224, 3] |
| **Output Shape** | [1, 2] |
| **Framework** | TensorFlow Lite |

### Output Classes
| Index | Label | Meaning |
| :--- | :--- | :--- |
| 0 | `colon_aca` | Colon Adenocarcinoma (Malignant) |
| 1 | `colon_n` | Colon Normal (Benign) |

### Preprocessing Pipeline
1.  **Raw Image** (Camera / Gallery)
2.  **EXIF Orientation Correction** (`bakeOrientation`)
3.  **Bilinear Resize** → 224 × 224 px
4.  **RGB Channel Extraction**
5.  **Normalization** (Raw pixel values 0.0 – 255.0 scaled to model requirements)
6.  **Input Tensor** `[1, 224, 224, 3]`
7.  **TFLite Inference** (INT8 MobileNetV2)
8.  **Raw Logits** `[1, 2]`
9.  **Softmax Normalization** (Flutter-side)
10. **Probabilities** (always summing to 100%)
11. **Argmax** → Predicted Class + Confidence

### OOD Detection Logic
**Stage 1 — Color Profiling (pre-inference):**
*   Sample 1000 random pixels from raw image
*   Check for pink (eosin) and purple (hematoxylin) color ratios
*   If < 20% of non-background pixels match H&E palette → REJECT immediately
*   *(catches photos, objects, screenshots, non-tissue images)*

**Stage 2 — Confidence Thresholding (post-inference):**
*   `topScore < 0.58` → "Not a colon slide" rejection
*   `topScore < 0.72` → Low confidence warning, pathologist review flagged
*   `topScore >= 0.72` → Valid result displayed

## Tech Stack
| Layer | Technology |
| :--- | :--- |
| **Frontend** | Flutter 3.x / Dart |
| **ML Inference** | TensorFlow Lite (`tflite_flutter ^0.12.1`) |
| **Image Processing** | `image ^4.9.2` |
| **State Management** | Provider `^6.1.2` |
| **Persistence** | `shared_preferences ^2.3.2` |
| **Camera / Gallery** | `image_picker ^1.1.2` |
| **Android SDK** | API Level 36 |
| **Android NDK** | Side-by-side r27 |
| **Build Tools** | Gradle 9.3.1 |
| **Training Dataset** | LC25000 (Colon subset) |
| **Model Training** | Keras / TensorFlow → TFLite conversion |

## Local Setup and Installation

### Prerequisites
*   Flutter SDK 3.x installed and on PATH
*   Android SDK with API 36
*   Android NDK installed (side-by-side)
*   A connected Android device with USB debugging enabled, or an emulator

### Steps
```bash
# Clone the repository
git clone [https://github.com/Muhammad-Bilal-998/MicroAI.git](https://github.com/Muhammad-Bilal-998/MicroAI.git)

# Navigate into the project directory
cd MicroAI

# Fetch Flutter dependencies
flutter pub get

# Verify your connected device
flutter devices

# Run the app on your connected device
flutter run

# Or build a release APK
flutter build apk --release --split-per-abi
