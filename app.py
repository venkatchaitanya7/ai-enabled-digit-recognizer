import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, render_template, request, jsonify
from model import DigitCNN

from PIL import ImageFile
ImageFile.Load_TRUNCATED_IMAGES = True

app = Flask(__name__)

# Load trained AI model
device = torch.device('cpu')
model = DigitCNN().to(device)
model.load_state_dict(torch.load('digit_model.pth', map_location=device))
model.eval()

def preprocess_image(image_base64, is_camera=False):
    image_bytes = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    if is_camera:
        # 1. Smooth out camera noise
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # 2. Local adaptive thresholding: subtract blurred background to isolate dark ink
        np_img = np.array(image, dtype=np.float32)
        bg = np.array(image.filter(ImageFilter.BoxBlur(21)), dtype=np.float32)
        diff = bg - np_img  # High positive values = dark ink on light paper

        # Dynamic contrast cutoff
        std_val = np.std(diff)
        thresh = max(18.0, np.mean(diff) + 0.8 * std_val)
        binary = np.where(diff > thresh, 255, 0).astype(np.uint8)
        image = Image.fromarray(binary)

        # 3. Morphological dilation: thicken thin pen strokes so CNN can see them clearly
        image = image.filter(ImageFilter.MaxFilter(5))

    # 4. Find bounding box of the digit
    bbox = image.getbbox()
    if bbox is None:
        return None, None

    cropped = image.crop(bbox)
    w, h = cropped.size
    if w < 6 or h < 6:
        return None, None

    # 5. Fit inside 20x20 box preserving aspect ratio
    aspect = w / h
    if aspect > 1:
        new_w = 20
        new_h = max(1, int(round(20 / aspect)))
    else:
        new_h = 20
        new_w = max(1, int(round(20 * aspect)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6. Center on 28x28 black canvas
    canvas = Image.new('L', (28, 28), color=0)
    canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))

    # Export debug base64 image so the user can see what the AI sees
    buffered = io.BytesIO()
    canvas.save(buffered, format="PNG")
    debug_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

    # 7. Normalize tensor
    img_array = np.array(canvas, dtype=np.float32) / 255.0
    normalized = (img_array - 0.1307) / 0.3081
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
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
    tensor, debug_img = preprocess_image(data['image'], is_camera=is_camera)

    if tensor is None:
        return jsonify({'error': 'No clear digit found. Write bolder or move closer to the camera!'}), 400

    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        confidence = float(probabilities[prediction] * 100)

    return jsonify({
        'prediction': prediction,
        'confidence': f"{confidence:.1f}%",
        'debug_image': debug_img
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
