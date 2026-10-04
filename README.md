<p align="center">
  <img src="https://img.icons8.com/fluency/96/hand-with-pen.png" alt="Character Recognition Logo" width="96"/>
</p>

<h1 align="center">Handwritten Character Recognition</h1>
<h3 align="center"><em>Real-time prediction of handwritten letters and digits.</em></h3>

<p align="center">
  An end-to-end Machine Learning web application that classifies handwritten digits and English letters (62 distinct classes) in real time using a Convolutional Neural Network (CNN) and an interactive HTML5 drawing canvas.
</p>

<p align="center">
  <a href="https://character-recognition-pe7v.onrender.com/">
    <img src="https://img.shields.io/badge/🌐_Live_Demo-character--recognition-4CAF50?style=for-the-badge" alt="Live Demo"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Frontend-Vanilla%20JS-F7DF1E?logo=javascript&logoColor=black" alt="Vanilla JS"/>
  <img src="https://img.shields.io/badge/Backend-Flask-000000?logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/ML-TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Infra-Docker-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Hosting-Render-46E3B7?logo=render&logoColor=white" alt="Render"/>
</p>

---

## 📖 About

**Handwritten Character Recognition** is a web-based tool that translates your handwriting into digital text. You draw a character on the canvas, hit predict, and the model tells you what it sees — along with a confidence score.

The background color of the page dynamically changes based on the prediction's confidence (green for high, orange for medium, red for low), providing immediate visual feedback.

> 🔗 **Try it live:** [character-recognition-pe7v.onrender.com](https://character-recognition-pe7v.onrender.com/)
>
> *Note: The free-tier Render instance may take ~30 seconds to spin up on the first visit.*

---

## 📸 Screenshots

### Drawing Canvas

<p align="center">
  <img src="screenshots/draw_before_predict.png" alt="Draw Before Predict" width="750"/>
</p>

The main interface features a responsive HTML5 drawing canvas with a custom pencil cursor. Users can draw any digit or letter (uppercase or lowercase). The UI features a modern glassmorphism design.

### High Confidence Prediction

<p align="center">
  <img src="screenshots/predict_high_confidence.png" alt="Predict High Confidence" width="750"/>
</p>

Once the 'Predict' button is pressed, the drawing is sent to the backend, preprocessed, and fed to the CNN. Here, the letter "A" is predicted with a very high confidence of 99.95%, triggering a green background and displaying the "Confidence Legend".

### Ambiguous Characters / Low Confidence

<p align="center">
  <img src="screenshots/predict_low_confidence.png" alt="Predict Low Confidence" width="750"/>
</p>

Some symbols inherently look alike and are ambiguous without context. For example, the digit `0` can easily be mistaken for the uppercase `O` or lowercase `o`, and vice versa. In these cases, the model's confidence naturally drops (e.g., 52.81%), and the UI reflects this uncertainty with an orange or red background.

---

## ✨ Highlights

| Aspect | Details |
|---|---|
| 🧠 **62-Class CNN** | Recognizes digits (0–9), uppercase (A–Z), and lowercase (a–z) letters from a single unified model |
| 🎨 **Interactive Canvas** | HTML5 drawing surface with custom pencil cursor and keyboard shortcuts for quick clear/predict |
| 📊 **Confidence Feedback** | Real-time color-coded background (green / orange / red) reflecting prediction certainty |
| 🔄 **In-Graph Augmentation** | Data augmentation layers (rotation, zoom, translation) baked directly into the model graph |
| 🐳 **Dockerized Deployment** | Production-ready containerization with Gunicorn, deployed on Render |
| 🖥️ **CLI Inference** | Predict characters from saved images without running the web server |

---

## 🏗️ Architecture & Preprocessing

```text
┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
│     HTML5 Frontend      │──────▶│      Flask Backend      │──────▶│     TensorFlow CNN      │
│                         │       │                         │       │                         │
│  • Canvas Drawing       │       │  • Image Preprocessing  │       │  • 3 Conv2D Blocks      │
│  • Base64 Image Export  │       │  • Background Composite │       │  • BatchNorm + Pooling   │
│  • Dynamic UI Colors    │◀──────│  • Confidence Scoring   │◀──────│  • Dense + Softmax (62) │
└─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
```

Getting a web canvas drawing to match the model's training data formatting takes a few crucial steps. Getting this wrong is the primary reason models fail in production:

1. **Background composite:** The canvas sends a PNG with transparency. We composite it onto a white background.
2. **Grayscale:** Convert the image to grayscale.
3. **Transpose:** EMNIST images are natively stored transposed (columns and rows swapped). Real-world drawings must be transposed to match this orientation.
4. **Invert:** Humans draw black strokes on white canvases, but EMNIST uses white strokes on a black background.
5. **Resize & Normalize:** Resize to 28×28 pixels, normalize pixel values to `[0, 1]`, and reshape to `(1, 28, 28, 1)`.

---

## 🧠 Model Structure

The model is built using TensorFlow/Keras and trained on the **EMNIST ByClass** dataset (~814,255 images, 62 classes).

```text
Input (28×28×1)
    │
    ├── RandomRotation(0.1)                   ┐
    ├── RandomZoom(0.1)                       ├── In-graph data augmentation
    ├── RandomTranslation(0.1, 0.1)           ┘
    │
    ├── Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2)     Block 1
    ├── Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2)     Block 2
    ├── Conv2D(128, 3×3, ReLU, padding=same) → BatchNorm      Block 3
    │
    ├── Flatten
    ├── Dense(256, ReLU) → BatchNorm → Dropout(0.5)
    └── Dense(62, Softmax) → Output
```

### Why a CNN?

We chose a **Sequential Convolutional Neural Network** as the core architecture for this project. CNNs are the dominant paradigm for image classification tasks because their convolutional filters are designed to learn **spatial hierarchies** — from low-level features (edges, strokes) to high-level concepts (loops, curves, character shapes). This makes them a natural fit for handwritten character recognition on 28×28 grayscale images. Below is a comparison against other approaches that were considered:

| Approach | How It Works | Why We Didn't Choose It |
|---|---|---|
| **MLP (Fully Connected)** | Flattens the 28×28 image into a 784-element vector and feeds it through dense layers | Destroys all spatial structure. Every pixel is treated independently, so the network cannot learn local patterns like edges or curves. Typically reaches ~75–80% on EMNIST ByClass — well below CNN performance. |
| **K-Nearest Neighbors (KNN)** | Compares a new image to every training image using pixel-distance metrics | Extremely slow at inference on 814K training samples (no learned representation). Sensitive to noise and translation shifts. Impractical for a real-time web application. |
| **SVM (Support Vector Machine)** | Finds optimal hyperplanes to separate classes in a high-dimensional feature space | Requires manual feature extraction (e.g., HOG descriptors). Does not natively learn hierarchical features. Scales poorly to 62 classes and hundreds of thousands of samples. |
| **RNN / LSTM** | Processes the image as a sequence of rows or columns | Designed for sequential data (text, time series). Treating image rows as timesteps is unnatural and fails to capture 2D spatial relationships. Slower to train with no accuracy benefit over CNNs for this task. |
| **Vision Transformer (ViT)** | Splits the image into patches and applies self-attention across all patches | Requires very large datasets (millions of images) and significant compute to outperform CNNs. Overkill for 28×28 images — the self-attention mechanism adds overhead without meaningful benefit at this resolution. The model would also be far too large for a lightweight web deployment. |
| ✅ **CNN (our choice)** | Applies learnable convolutional filters that slide across the image, capturing local spatial patterns at multiple scales | Natively preserves spatial structure. Efficiently learns edge → shape → character hierarchies. Small model footprint (~11 MB). Fast inference suitable for real-time web use. Achieves **~85.5% accuracy** on 62 classes — strong given the inherent ambiguity between visually similar characters (0/O/o, 1/l/I, etc.). |

### Key Design Decisions

- **3 Convolutional Blocks (32 → 64 → 128 filters):** A progressive filter increase lets the network learn increasingly complex features at each level. More blocks were tested but added parameters without improving accuracy, while fewer blocks under-fitted on the 62-class problem.
- **Batch Normalization after every Conv2D:** Stabilizes and accelerates training by normalizing activations. This was especially important given the large class count and class imbalance in EMNIST ByClass.
- **In-Graph Data Augmentation:** Random rotation (±10°), zoom (±10%), and translation (±10%) are applied as Keras layers *inside* the model graph. This means augmentation happens on-the-fly during training (not applied at inference), improving generalization without inflating the dataset size on disk.
- **50% Dropout:** Aggressive dropout before the final classifier prevents co-adaptation of neurons and reduces overfitting, which is critical when many classes share similar visual features.
- **Adam Optimizer with ReduceLROnPlateau:** The learning rate starts at the Adam default (1e-3) and is automatically reduced by 80% when validation loss plateaus, enabling fine-grained convergence in later epochs.

---

## 📈 Training Details & Results

| Parameter | Value |
|---|---|
| **Dataset** | [EMNIST ByClass](https://www.nist.gov/itl/products-and-services/emnist-dataset) — 814,255 images, 62 classes |
| **Batch Size** | 128 |
| **Optimizer** | Adam (initial LR = 1e-3) |
| **Loss Function** | Sparse Categorical Cross-Entropy |
| **Epochs** | 25 (with early stopping, patience = 4) |
| **LR Schedule** | ReduceLROnPlateau (factor = 0.2, patience = 2, min_lr = 1e-5) |
| **Best Validation Accuracy** | **~85.5%** |
| **Best Validation Loss** | **~0.383** |
| **Model File Size** | ~11.2 MB (`.h5` format) |

> **Note on accuracy:** A ~85.5% accuracy across 62 classes might seem modest compared to the 99%+ scores seen on MNIST (10 digits only). However, EMNIST ByClass includes inherently ambiguous character pairs — `0`/`O`/`o`, `1`/`l`/`I`, `S`/`s`, `C`/`c`, `V`/`v`, `W`/`w`, etc. — where even humans disagree. The confusion matrix in the [training notebook](notebooks/best-model-building.ipynb) confirms that the vast majority of misclassifications occur between these visually identical pairs, not between genuinely different characters.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | [HTML5 Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) & JS | Interactive drawing interface |
| **Backend** | [Flask](https://flask.palletsprojects.com/) | REST API and web server |
| **Machine Learning** | [TensorFlow / Keras](https://www.tensorflow.org/) | CNN architecture, training, and inference |
| **Data Processing** | [Pillow (PIL)](https://python-pillow.org/) & [NumPy](https://numpy.org/) | Image manipulation and array formatting |
| **Dataset** | [EMNIST ByClass](https://www.nist.gov/itl/products-and-services/emnist-dataset) | 62-class extended MNIST dataset |
| **Containerization** | [Docker](https://www.docker.com/) | Reproducible deployment environment |
| **WSGI Server** | [Gunicorn](https://gunicorn.org/) | Production-grade Python HTTP server |
| **Hosting** | [Render](https://render.com/) | Cloud deployment with free-tier support |

---

## 📁 Repository Structure

```text
character-recognition/
├── src/
│   ├── model.py                    # CNN architecture definition
│   ├── train.py                    # Training script & dataset pipeline
│   └── predict.py                  # CLI prediction script
├── templates/
│   └── index.html                  # Main web application template
├── static/
│   ├── css/style.css               # Glassmorphism styling and responsive layout
│   ├── js/main.js                  # Canvas logic, prediction requests, and shortcuts
│   └── images/pencil-cursor.png    # Custom drawing cursor
├── notebooks/
│   └── best-model-building.ipynb   # Jupyter notebook with experimentation, training, and confusion matrix
├── screenshots/                    # Application preview images
├── app.py                          # Flask backend server
├── emnist_cnn_62_class_v1.h5       # Pre-trained CNN model weights (~11 MB)
├── Dockerfile                      # Containerization setup
├── requirements.txt                # Python dependencies
└── data/                           # Directory for local dataset caching
```

---

## 🚀 Getting Started

### Live Demo

The easiest way to try the app is through the live deployment:

> **[https://character-recognition-pe7v.onrender.com/](https://character-recognition-pe7v.onrender.com/)**

### Local Setup

#### Prerequisites

- [Python 3.8+](https://www.python.org/)
- `pip`

#### Installation

```bash
# Clone the repository
git clone https://github.com/nikolasfragkos/character-recognition.git
cd character-recognition

# Install dependencies
pip install -r requirements.txt
pip install flask pillow

# Run the app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### Docker

```bash
# Build the image
docker build -t character-recognition .

# Run the container
docker run -p 7860:7860 character-recognition
```

Open `http://localhost:7860` in your browser.

### Command Line Interface

You can also predict from a saved image without running the web server:

```bash
python src/predict.py --image path/to/your/image.png
```

---

## 🌐 Deployment

The application is deployed on **[Render](https://render.com/)** as a Docker-based web service.

| Detail | Value |
|---|---|
| **Platform** | Render (Free Tier) |
| **URL** | [character-recognition-pe7v.onrender.com](https://character-recognition-pe7v.onrender.com/) |
| **Container** | Docker (Python 3.9 slim) |
| **WSGI Server** | Gunicorn |
| **Port** | 7860 |

> **Cold starts:** Free-tier Render instances spin down after periods of inactivity. The first request after a cold start may take ~30 seconds while the container boots and the TensorFlow model loads into memory.