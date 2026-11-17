import io
import base64
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageFile
from flask import Flask, render_template, request, jsonify

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

# Load Model: Try ONNX first, otherwise fallback to PyTorch
USE_ONNX = False
try:
    import onnxruntime as ort
    ort_session = ort.InferenceSession("digit_model.onnx")
    USE_ONNX = True
    print("✅ Loaded ONNX Engine (Memory: ~35MB RAM)")
except Exception as e:
    print(f"⚠️ ONNX failed to load, falling back to PyTorch: {e}")
    import torch
    import torch.nn.functional as F
    from model import DigitCNN
    device = torch.device('cpu')
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load('digit_model.pth', map_location=device))
    model.eval()

def get_center_of_mass(img_array):
    """Calculates the center of mass of the pixels (MNIST standard)."""
    cy, cx = np.indices(img_array.shape)
    total = img_array.sum()
    if total == 0:
        return 14, 14
    x_center = (cx * img_array).sum() / total
    y_center = (cy * img_array).sum() / total
    return x_center, y_center

def shift_image(img_array, dx, dy):
    shifted = np.zeros_like(img_array)
    h, w = img_array.shape
    x_start = max(0, int(dx))
    x_end = min(w, w + int(dx))
    y_start = max(0, int(dy))
    y_end = min(h, h + int(dy))
    src_x_start = max(0, -int(dx))
    src_x_end = min(w, w - int(dx))
    src_y_start = max(0, -int(dy))
    src_y_end = min(h, h - int(dy))
    shifted[y_start:y_end, x_start:x_end] = img_array[src_y_start:src_y_end, src_x_start:src_x_end]
    return shifted

def preprocess_image(image_base64, is_photo=False):
    image_bytes = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    if is_photo:
        # 1. Clean background shadows
        np_img = np.array(image, dtype=np.float32)
        bg = np.array(image.filter(ImageFilter.BoxBlur(25)), dtype=np.float32)
        diff = bg - np_img
        thresh = max(16.0, np.mean(diff) + 0.7 * np.std(diff))
        binary = np.where(diff > thresh, 255, 0).astype(np.uint8)
        image = Image.fromarray(binary)

        # 2. Thicken pen strokes
        image = image.filter(ImageFilter.MaxFilter(5))

    # 3. Crop content
    bbox = image.getbbox()
    if bbox is None:
        return None, None

    cropped = image.crop(bbox)
    w, h = cropped.size
    if w < 6 or h < 6:
        return None, None

    # 4. Fit inside 20x20
    aspect = w / h
    if aspect > 1:
        new_w = 20
        new_h = max(1, int(round(20 / aspect)))
    else:
        new_h = 20
        new_w = max(1, int(round(20 * aspect)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 5. Center on 28x28 black canvas
    canvas = Image.new('L', (28, 28), color=0)
    canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))

    img_array = np.array(canvas, dtype=np.float32)

    # 6. Center of Mass Shift
    cx, cy = get_center_of_mass(img_array)
    shift_x = 14.0 - cx
    shift_y = 14.0 - cy
    if abs(shift_x) < 5 and abs(shift_y) < 5:
        img_array = shift_image(img_array, shift_x, shift_y)

    preview_img = Image.fromarray(img_array.astype(np.uint8))
    buffered = io.BytesIO()
    preview_img.save(buffered, format="PNG")
    debug_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

    # 7. Normalize
    normalized = (img_array / 255.0 - 0.1307) / 0.3081
    tensor = normalized.reshape(1, 1, 28, 28).astype(np.float32)

    return tensor, debug_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    is_camera = data.get('is_camera', False)
    tensor, debug_img = preprocess_image(data['image'], is_photo=is_camera)

    if tensor is None:
        return jsonify({'error': 'No clear digit found!'}), 400

    if USE_ONNX:
        outputs = ort_session.run(None, {'input': tensor})[0][0]
        exp = np.exp(outputs - np.max(outputs))
        probabilities = exp / exp.sum()
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction] * 100)
    else:
        # PyTorch Fallback Prediction
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            out = model(torch.tensor(tensor))
            prob = F.softmax(out, dim=1)[0]
            prediction = torch.argmax(prob).item()
            confidence = float(prob[prediction] * 100)

    return jsonify({
        'prediction': prediction,
        'confidence': f"{confidence:.1f}%",
        'debug_image': debug_img
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
