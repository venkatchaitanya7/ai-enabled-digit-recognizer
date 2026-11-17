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
| **Model** | Custom CNN (2 conv layers + 2 FC layers) |
| **Training Data** | MNIST (60,000 train / 10,000 test) |
| **Frontend** | Vanilla HTML/CSS/JavaScript |
| **Deployment** | Render (gunicorn) |

## Model Architecture

```
Input: 1×28×28 (grayscale)
├── Conv2d(1→32, 3×3, padding=1) + ReLU
├── MaxPool2d(2×2)
├── Conv2d(32→64, 3×3, padding=1) + ReLU
├── MaxPool2d(2×2) → 64×7×7
├── Dropout(0.25)
├── Flatten → 3136
├── Linear(3136→128) + ReLU
├── Dropout(0.5)
└── Linear(128→10) → Logits (0-9)
```

- **Accuracy**: ~99% on MNIST test set
- **Training**: 8 epochs with data augmentation (rotation, translation, scaling)
- **Normalization**: MNIST mean=0.1307, std=0.3081

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
├── app.py                 # Flask application with /predict endpoint
├── model.py               # PyTorch CNN model definition
├── train.py               # Training script with data augmentation
├── requirements.txt       # Python dependencies
├── render.yaml            # Render.com deployment config
├── digit_model.pth        # Pre-trained model weights (1.7 MB)
├── .gitignore
└── templates/
    └── index.html         # Frontend (HTML/CSS/JS all-in-one)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main page with drawing canvas, camera, upload |
| `POST` | `/predict` | Digit prediction |
| `GET` | `/health` | Health check (optional) |

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

## Image Preprocessing Pipeline

The model expects MNIST-format images (28×28, white digit on black background, centered). The preprocessing handles real-world inputs:

### For Canvas Drawings (white on black)
1. Normalize to 0-1 range
2. Find bounding box of strokes
3. Crop → Resize to 20×20 (aspect preserved)
4. Center on 28×28 canvas
5. Center-of-mass alignment
6. Normalize with MNIST stats

### For Camera/Upload (pen on paper, varying lighting)
1. Median filter (denoise)
2. Adaptive background subtraction (BoxBlur - original)
3. Dynamic threshold: `mean + 0.8×std` (min 18)
4. MaxFilter dilation (thicken strokes)
5. Same geometric steps as canvas

## Development

### Retraining the Model
```bash
python train.py
# Trains for 8 epochs with augmentation
# Saves to digit_model.pth
```

### Model Performance
```
Epoch 01/08 | Loss: 0.2156 | Test Accuracy: 92.34%
Epoch 02/08 | Loss: 0.0892 | Test Accuracy: 97.12%
Epoch 03/08 | Loss: 0.0621 | Test Accuracy: 97.85%
Epoch 04/08 | Loss: 0.0483 | Test Accuracy: 98.21%
Epoch 05/08 | Loss: 0.0392 | Test Accuracy: 98.45%
Epoch 06/08 | Loss: 0.0328 | Test Accuracy: 98.62%
Epoch 07/08 | Loss: 0.0281 | Test Accuracy: 98.78%
Epoch 08/08 | Loss: 0.0245 | Test Accuracy: 98.89%
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Run `python train.py` first |
| Camera not working | Use HTTPS (required for `getUserMedia`), allow camera permission |
| Low accuracy on camera | Ensure good lighting, write bold digits, center in reticle |
| Render deploy fails | Check build logs; ensure `digit_model.pth` isn't in `.gitignore` |

## License

MIT License - Feel free to use for learning, teaching, or commercial projects.

## Credits

- MNIST dataset: Yann LeCun, Corinna Cortes, Christopher Burges
- PyTorch & torchvision teams
- Inspired by classic digit recognition tutorials