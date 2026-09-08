import os
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client

# --- CONFIGURATION (Pre-filled for you) ---
EMAIL_USER = "kunalz2308@gmail.com"
EMAIL_PASS = "ucbr bekh dsgg pakc"
DOCTOR_EMAIL = "himanshutapde.skn.comp@gmail.com"

# Twilio Configuration
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
WHATSAPP_FROM = "whatsapp:+14155238886"

def trigger_alert(disease, confidence, patient_data, verify_link):
    # Logic: Alert if it is TB/Cancer OR confidence is very high (>85%)
    severity = "HIGH" if disease in ["Tuberculosis", "Lung_Cancer"] or confidence > 85 else "LOW"
    logs = []

    # 1. SEND EMAIL
    try:
        msg_body = f"""
        PULMOSIGHT MEDICAL ALERT
        ------------------------
        Diagnosis: {disease}
        Confidence: {confidence}%
        Patient Data: {patient_data}
        
        VERIFY REPORT HERE:
        {verify_link}
        """
        msg = MIMEText(msg_body)
        msg['Subject'] = f"[{severity}] PulmoSight Alert: {disease}"
        msg['From'] = EMAIL_USER
        msg['To'] = DOCTOR_EMAIL
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logs.append("Email Sent")
    except Exception as e:
        logs.append(f"Email Error: {e}")

    # 2. SEND WHATSAPP (Critical Alerts Only)
    if severity == "HIGH":
        try:
            client = Client(TWILIO_SID, TWILIO_TOKEN)
            # Use the doctor's phone number if provided, otherwise default to yours for the demo
            target_phone = "whatsapp:+917875138495" # <--- REPLACE WITH YOUR PHONE NUMBER (+91...)
            
            client.messages.create(
                body=f"🚨 *CRITICAL ALERT*: {disease} detected ({confidence}%). Verify immediately: {verify_link}",
                from_=WHATSAPP_FROM,
                to=target_phone
            )
            logs.append("WhatsApp Sent")
        except Exception as e:
            logs.append(f"WhatsApp Error: {e}")

    return ", ".join(logs)
