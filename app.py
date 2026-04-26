import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64

app = Flask(__name__)

# ── Load model once at startup ─────────────────────────────────
MODEL_PATH = "model/alzheimer_cnn_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully - Ready for 75.71% Accuracy Inference")

IMG_SIZE    = 224
CLASS_NAMES = ["Normal", "Mild", "Alzheimer's"]
CLASS_INFO  = {
    "Normal":       {"color": "#34d399", "risk": "Low",    "desc": "No significant signs of neurodegeneration detected. Brain structures appear within normal range for age."},
    "Mild":         {"color": "#fbbf24", "risk": "Medium", "desc": "Subtle signs of cortical thinning or ventricular enlargement detected. Early-stage follow-up recommended."},
    "Alzheimer's":  {"color": "#f87171", "risk": "High",   "desc": "Significant hippocampal atrophy and ventricular enlargement detected. Clinical intervention required."},
}

def preprocess_image(img_bytes):
    """
    Matches the Colab 'preprocess_image_tf' exactly.
    This ensures the 75.71% test accuracy is maintained in production.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG or PNG.")

    # 1. Resize to match training
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 2. Convert BGR to RGB (Training was done in RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Normalize (Matches: img / 255.0)
    processed = img_rgb.astype(np.float32) / 255.0

    return np.expand_dims(processed, axis=0)

def generate_gradcam(img_bytes, pred_class_idx):
    """
    Generate Grad-CAM heatmap using the same RGB/255.0 preprocessing.
    Removed EQ/Blur to match the model's training distribution.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not decode image")

    # Match preprocessing exactly
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_f32 = img_rgb.astype(np.float32) / 255.0

    # Build Grad-CAM model (Targeting the last conv layer of ResNet50)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("conv5_block3_out").output, model.output]
    )

    inp = tf.expand_dims(img_f32, 0)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp, training=False)
        class_score = preds[:, pred_class_idx]

    grads = tape.gradient(class_score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Heatmap computation
    heatmap = (conv_out[0] @ tf.expand_dims(pooled, axis=-1)).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # Create overlay
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_c = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay on the RGB resized image
    overlay = cv2.addWeighted(img_rgb.astype(np.uint8), 0.6, 
                              cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB), 0.4, 0)

    # Encode to base64
    pil_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    try:
        # Step 1: Preprocess
        inp = preprocess_image(img_bytes)
        
        # Step 2: Inference
        probs = model.predict(inp, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        # Step 3: Probabilities for the UI Chart
        probabilities = [
            {"class": CLASS_NAMES[i], "probability": round(float(probs[i]) * 100, 2)}
            for i in range(len(CLASS_NAMES))
        ]

        # Step 4: Explainability (Grad-CAM)
        gradcam_img = generate_gradcam(img_bytes, pred_idx)

        return jsonify({
            "prediction":     pred_class,
            "confidence":     round(confidence, 2),
            "probabilities":  probabilities,
            "gradcam":        gradcam_img,
            "color":          CLASS_INFO[pred_class]["color"],
            "risk":           CLASS_INFO[pred_class]["risk"],
            "description":    CLASS_INFO[pred_class]["desc"],
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)