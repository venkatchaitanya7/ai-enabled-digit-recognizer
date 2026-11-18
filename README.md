# AI-Enabled Digit Recognizer

A Flask-based web application for real-time handwritten digit recognition using a Convolutional Neural Network (CNN) trained on MNIST. Draw digits on a canvas, capture from webcam, or upload images for instant prediction.

**Live Demo:** https://ai-enabled-digit-recognizer.onrender.com/

## Features

- ✏️ **Digital Canvas** - Draw digits with mouse or touch (desktop & mobile)
- 📷 **Webcam Capture** - Real-time camera feed with reticle for digit capture
- 📤 **Image Upload** - Drag & drop or click to upload images (PNG, JPG up to 5MB)
- 🔮 **Real-time Prediction** - CNN model with confidence scores and debug view
- 🎨 **Modern UI** - Dark theme with cyan accents, fully responsive

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask (Python) |
| **ML Framework** | PyTorch |
| **Model** | Custom CNN (4 conv layers + 2 FC layers) |
| **Training Data** | MNIST (60,000 train / 10,000 test) |
| **Frontend** | Vanilla HTML/CSS/JavaScript |
| **Deployment** | Render, Heroku, Railway, Fly.io, Docker |

## Model Architecture

```
Input: 1×28×28 (grayscale)
├── Block 1: Conv2d(1→32, 3×3, padding=1) + BN + ReLU
│           Conv2d(32→32, 3×3, padding=1) + BN + ReLU
│           MaxPool2d(2×2) + Dropout2d(0.25)
├── Block 2: Conv2d(32→64, 3×3, padding=1) + BN + ReLU
│           Conv2d(64→64, 3×3, padding=1) + BN + ReLU
│           MaxPool2d(2×2) + Dropout2d(0.25) → 64×7×7
├── Flatten → 3136
├── Linear(3136→256) + BN + ReLU + Dropout(0.4)
└── Linear(256→10) → Logits (0-9)
```

- **Accuracy**: ~99% on MNIST test set (after 10 epochs with augmentation)
- **Training**: 10 epochs with data augmentation (rotation, translation, scaling)
- **Normalization**: MNIST mean=0.1307, std=0.3081
- **Export**: ONNX format (~35 MB) for ultra-fast inference

## Quick Start

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/venkatchaitanya7/ai-enabled-digit-recognizer.git
cd ai-enabled-digit-recognizer

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (or use pre-trained digit_model.pth)
python train.py

# 5. Run the Flask app
python app.py

# 6. Open http://localhost:5000
```

### Deploy to Render (Free Tier)

1. **Fork/Clone** this repository to your GitHub
2. **Connect to Render**: New → Web Service → Connect your repo
3. **Render auto-detects** `render.yaml` and configures:
   - Build: `pip install -r requirements.txt && python train.py`
   - Start: `gunicorn app:app`
   - Python: 3.11
4. **Deploy** - First deploy trains the model (~2-3 min), subsequent deploys are faster

> ⚠️ **Note**: Render free tier spins down after 15 min inactivity. First request after wake-up takes 30-60 seconds.

## Project Structure

```
ai-enabled-digit-recognizer/
├── app.py                 # Flask application with /predict & /health endpoints
├── model.py               # PyTorch CNN model definition
├── train.py               # Training script with data augmentation
├── requirements.txt       # Python dependencies (pinned)
├── render.yaml            # Render.com deployment config
├── Procfile               # Heroku deployment config
├── railway.toml           # Railway deployment config
├── fly.toml               # Fly.io deployment config
├── Dockerfile             # Container deployment
├── docker-compose.yml     # Local container development
├── .dockerignore
├── .gitignore
├── runtime.txt            # Python version for Heroku
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI/CD
├── digit_model.pth        # Pre-trained model weights (1.7 MB) - Git LFS recommended
└── templates/
    └── index.html         # Frontend (HTML/CSS/JS all-in-one)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main page with drawing canvas, camera, upload |
| `POST` | `/predict` | Digit prediction |
| `GET` | `/health` | Health check for load balancers |

### `/predict` Request

```json
{
  "image": "data:image/png;base64,...",
  "is_camera": false
}
```

### `/predict` Response

```json
{
  "prediction": 7,
  "confidence": "98.5%",
  "debug_image": "data:image/png;base64,..."
}
```

### `/health` Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "onnx|pytorch"
}
```

## Image Preprocessing Pipeline

The model expects MNIST-format images (28×28, white digit on black background, centered). The preprocessing handles real-world inputs:

### For Canvas Drawings (white on black)
1. Normalize to 0-1 range
2. Find bounding box of strokes
3. Crop → Resize to 20×20 (aspect preserved)
4. Center on 28×28 canvas
5. Center-of-mass alignment
6. Normalize with MNIST stats (mean=0.1307, std=0.3081)

### For Camera/Upload (pen on paper, varying lighting)
1. BoxBlur background estimation (radius=25)
2. Adaptive background subtraction (diff = bg - img)
3. Dynamic threshold: `mean + 0.7×std` (min 16)
4. MaxFilter dilation (size=5, thicken strokes)
5. Same geometric steps as canvas

## Development

### Retraining the Model

```bash
python train.py
# Trains for 10 epochs with augmentation
# Saves to digit_model.pth and digit_model.onnx
```

### Model Performance (Typical)

```
Epoch 01/10 | Loss: 0.2156 | Test Accuracy: 92.34%
Epoch 02/10 | Loss: 0.0892 | Test Accuracy: 97.12%
Epoch 03/10 | Loss: 0.0621 | Test Accuracy: 97.85%
Epoch 04/10 | Loss: 0.0483 | Test Accuracy: 98.21%
Epoch 05/10 | Loss: 0.0392 | Test Accuracy: 98.45%
Epoch 06/10 | Loss: 0.0328 | Test Accuracy: 98.62%
Epoch 07/10 | Loss: 0.0281 | Test Accuracy: 98.78%
Epoch 08/10 | Loss: 0.0245 | Test Accuracy: 98.89%
Epoch 09/10 | Loss: 0.0218 | Test Accuracy: 98.95%
Epoch 10/10 | Loss: 0.0196 | Test Accuracy: 99.02%
```

## Deployment Guides

### Render (Recommended - Free Tier)

See `render.yaml` - automatic detection, builds ONNX model at build time.

### Heroku

```bash
# Install Heroku CLI, then:
heroku create your-app-name
heroku buildpacks:set heroku/python
git push heroku main
```

Uses `Procfile` and `runtime.txt`.

### Railway

```bash
railway login
railway init
railway up
```

Uses `railway.toml`.

### Fly.io

```bash
fly launch --config fly.toml
fly deploy
```

Uses `fly.toml` (Docker-based).

### Docker (Any Platform)

```bash
docker build -t digit-recognizer .
docker run -p 8000:8000 digit-recognizer
```

Or with compose:
```bash
docker-compose up --build
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Run `python train.py` first |
| Camera not working | Use HTTPS (required for `getUserMedia`), allow camera permission |
| Low accuracy on camera | Ensure good lighting, write bold digits, center in reticle |
| Render deploy fails | Check build logs; ensure `digit_model.pth` isn't in `.gitignore` |
| ONNX load fails | Falls back to PyTorch automatically; check `onnxruntime` install |
| Memory errors on free tier | ONNX model is ~35MB; ensure 512MB+ RAM available |

## License

MIT License - Feel free to use for learning, teaching, or commercial projects.

## Credits

- MNIST dataset: Yann LeCun, Corinna Cortes, Christopher Burges
- PyTorch & torchvision teams
- Inspired by classic digit recognition tutorials