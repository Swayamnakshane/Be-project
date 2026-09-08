import os
import time
import json
import requests
import re
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input  # EfficientNetB0
from agent import trigger_alert
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY   = os.getenv("GROQ_API_KEY")
MODEL_PATH = "pulmosight_model.h5"
CLASSES    = ['Lung_Cancer', 'Normal', 'Pneumonia', 'Tuberculosis']

print("────────────────────────────────────────────────")
print("🧠 STARTING PULMOSIGHT ENGINE (COLLEGE VERSION)...")
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"   ✅ Model loaded: {MODEL_PATH}")
    print(f"   📐 Input shape : {model.input_shape}")
    print(f"   📊 Output units: {model.output_shape[-1]}")
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        print(f"   🔑 Gemini API  : configured")
    if GROQ_KEY:
        print(f"   🔑 Groq API    : configured")
    print("   🛡️  Safety net  : Lung_Cancer escalation @ >15%")
    print("✅ SYSTEM READY. Listening on Port 5000.")
except Exception as e:
    print(f"❌ FATAL: Could not load model — {e}")
print("────────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════
# PREDICTION ENGINE — TTA + Cancer Safety Net
# ═══════════════════════════════════════════════════════════════

def predict_with_tta(img_path):
    """
    Test-Time Augmentation: 5 views averaged for stability.
    Uses EfficientNetB0 preprocess_input.
    """
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)

    views = np.array([
        x,
        np.fliplr(x),
        np.clip(x * 1.10, 0, 255),
        np.clip(x * 0.90, 0, 255),
        np.fliplr(np.clip(x * 1.05, 0, 255)),
    ])

    views = preprocess_input(views.copy())
    preds = model.predict(views, verbose=0)
    return np.mean(preds, axis=0)


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS ROUTE
# ═══════════════════════════════════════════════════════════════

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # 1. SAVE IMAGE
    f = request.files['image']
    path = os.path.join("temp", f.filename)
    f.save(path)

    # 2. DETECT DISEASE
    try:
        avg_pred = predict_with_tta(path)

        print(f"📊 Predictions: { {CLASSES[i]: f'{avg_pred[i]*100:.1f}%' for i in range(len(CLASSES))} }")

        top_idx    = np.argmax(avg_pred)
        top_conf   = float(np.max(avg_pred))
        disease    = CLASSES[top_idx]

        sorted_idx  = np.argsort(avg_pred)[::-1]
        second_idx  = sorted_idx[1]
        second_conf = float(avg_pred[second_idx])
        margin      = top_conf - second_conf

        cancer_prob = float(avg_pred[CLASSES.index('Lung_Cancer')])

        # ── CLINICAL SAFETY NET ──
        # 15% threshold for college version (catches more cancer)
        if cancer_prob > 0.15 and disease != 'Lung_Cancer':
            disease = 'Lung_Cancer'
            conf = round(cancer_prob * 100, 2)
            print(f"🚨 SAFETY OVERRIDE: Lung_Cancer at {conf}% — escalating")
            print(f"   Original was {CLASSES[top_idx]} @ {top_conf*100:.1f}%")
        elif top_conf < 0.55 or margin < 0.15:
            conf = round(top_conf * 100, 2)
            print(f"⚠️  LOW CONFIDENCE: {disease} @ {conf}%")
        else:
            conf = round(top_conf * 100, 2)
            print(f"🎯 Result: {disease} @ {conf}%")

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"error": f"Model Error: {str(e)}"}), 500

    # 3. GENERATE REPORT (English Only — College Version)
    patient_data = request.form.to_dict()
    timestamp    = time.strftime("%d/%m/%Y")
    report_en    = ""
    report_hi    = ""

    prompt = f"""
    Act as a highly experienced Senior Radiologist. 
    Analyze this diagnosis: {disease}.
    
    Return ONLY a valid JSON object. Do NOT add markdown formatting.
    {{
        "trachea_en": "(Describe trachea/mediastinum findings)",
        "lungs_en": "(Describe lung/pleura opacities or clear fields)",
        "heart_en": "(Describe cardiac silhouette)",
        "impression_en": "(Clear definitive conclusion)",
        "recommendation_en": "(Actionable next steps)"
    }}
    """

    def format_report(data):
        return (f"RADIOLOGICAL FINDINGS:\n"
                f"• TRACHEA & MEDIASTINUM: {data.get('trachea_en', 'Normal')}\n"
                f"• LUNGS & PLEURA: {data.get('lungs_en', 'Normal')}\n"
                f"• HEART: {data.get('heart_en', 'Normal')}\n\n"
                f"IMPRESSION:\n"
                f"• {data.get('impression_en', 'Review required.')}\n"
                f"• {data.get('recommendation_en', 'Consult physician.')}")

    # TIER 1: GEMINI
    try:
        print("🌐 Attempting Tier 1: Gemini API...")
        if not GEMINI_KEY:
            raise ValueError("Gemini key missing")
        response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            report_en = format_report(data)
            print("✅ Gemini report generated.")
        else:
            raise ValueError("Invalid JSON")
    except Exception as e_gemini:
        print(f"⚠️ Gemini Failed: {e_gemini}")

        # TIER 2: GROQ
        try:
            print("🔄 Switching to Groq API...")
            if not GROQ_KEY:
                raise ValueError("Groq key missing")
            headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            response = requests.post(
                url="https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=30
            )
            if response.status_code != 200:
                raise Exception(f"Groq HTTP {response.status_code}")
            result = response.json()
            if 'error' in result:
                raise Exception(f"Groq Error: {result['error']}")
            ai_text = result['choices'][0]['message']['content']
            match = re.search(r'\{.*\}', ai_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                report_en = format_report(data)
                print("✅ Groq report generated.")
            else:
                raise ValueError("Invalid JSON")
        except Exception as e_groq:
            print(f"⚠️ Groq Failed: {e_groq}")
            print("🔌 Offline mode...")
            report_hi = ""
            fallback = {
                'Tuberculosis': "RADIOLOGICAL FINDINGS:\n• TRACHEA & MEDIASTINUM: Central. No mediastinal shift.\n• LUNGS & PLEURA: Heterogeneous opacities in upper lobes. Cavitary changes noted.\n• HEART: Normal cardiac size.\n\nIMPRESSION:\n• Findings suggestive of Pulmonary Tuberculosis.\n• Recommend Sputum AFB correlation.",
                'Pneumonia': "RADIOLOGICAL FINDINGS:\n• TRACHEA & MEDIASTINUM: Central.\n• LUNGS & PLEURA: Focal consolidation with air bronchograms.\n• HEART: Normal limits.\n\nIMPRESSION:\n• Lobar Pneumonia.\n• Recommend antibiotic therapy.",
                'Lung_Cancer': "RADIOLOGICAL FINDINGS:\n• TRACHEA & MEDIASTINUM: Deviated slightly due to mass effect.\n• LUNGS & PLEURA: Spiculated nodular opacity observed.\n• HEART: Normal.\n\nIMPRESSION:\n• Suspicious pulmonary mass. Malignancy cannot be ruled out.\n• CT Chest & Biopsy required immediately.",
                'Normal': "RADIOLOGICAL FINDINGS:\n• TRACHEA & MEDIASTINUM: Central.\n• LUNGS & PLEURA: Clear lung fields. No focal consolidation.\n• HEART: Normal cardiac silhouette.\n\nIMPRESSION:\n• Normal Chest X-Ray.\n• No acute cardiopulmonary abnormality seen."
            }
            report_en = fallback.get(disease, fallback['Normal'])

    # 4. ALERTS
    logs = trigger_alert(disease, conf, patient_data, "http://localhost:3000/verify")
    if os.path.exists(path):
        os.remove(path)

    return jsonify({
        "disease": disease,
        "confidence": conf,
        "report": report_en,
        "report_hindi": report_hi,  # Empty string for college version
        "agent_logs": logs
    })


# ═══════════════════════════════════════════════════════════════
# CHAT ROUTE
# ═══════════════════════════════════════════════════════════════

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data:
        return jsonify({"answer": "Invalid request."}), 400

    question       = data.get('question', '')
    report_context = data.get('report_context', '')

    prompt = f"""
    You are 'PulmoSight Clinical AI', a medical assistant. 
    Answer based STRICTLY on this report: {report_context}
    Question: {question}
    Keep it professional and concise.
    """

    try:
        if GEMINI_KEY:
            response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
            return jsonify({"answer": response.text.strip()})
        raise ValueError("No Gemini key")
    except:
        try:
            if GROQ_KEY:
                headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
                payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]}
                res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
                if res.status_code == 200:
                    result = res.json()
                    if 'error' not in result:
                        return jsonify({"answer": result['choices'][0]['message']['content'].strip()})
            raise Exception("APIs failed")
        except:
            return jsonify({"answer": "Offline mode. Please consult a specialist."})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "version": "College (EfficientNetB0)",
        "model_accuracy": "~90%",
        "safety_net": "Lung_Cancer escalation @ >15%",
        "classes": CLASSES
    })


if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run(port=5000, host='0.0.0.0')