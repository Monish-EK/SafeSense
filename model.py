import cv2
import os
import csv
import smtplib
import mimetypes
from email.mime.text import MIMEText
from email.message import EmailMessage
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from twilio.rest import Client

# Load the YOLOv8 model
model = YOLO("./Safety-Detection-YOLOv8/ppe.pt")  # Replace with your trained YOLO model

# Create a face recognizer using LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Gmail SMTP Configuration (✅ Free)
GMAIL_USER = "221701038@rajalakshmi.edu.in"
GMAIL_PASSWORD = "221701038"  # Use the App Password

# HR Email
HR_EMAIL = "monishdhanush75@gmail.com"

# Function to send email via Gmail SMTP
def send_email(to_email, subject, body, attachment_path=None):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg.set_content(body)

    # Attach image if available
    if attachment_path:
        with open(attachment_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
            mime_type, _ = mimetypes.guess_type(attachment_path)
            mime_main, mime_sub = mime_type.split('/')
            msg.add_attachment(file_data, maintype=mime_main, subtype=mime_sub, filename=file_name)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

# Function to send an email notification when safety violation occurs
def notify_hr(name, screenshot_path):
    subject = f"Safety Violation Alert: {name}"
    body = f"A safety violation was detected for {name}.\n\nSee attached image for details."
    send_email(HR_EMAIL, subject, body, screenshot_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Employee training
def train_face_recognizer(face_folder_path):
    faces, labels, label_map = [], [], {}
    current_label = 0

    for filename in os.listdir(face_folder_path):
        if filename.endswith(('.jpg', '.png')):
            image = cv2.imread(os.path.join(face_folder_path, filename))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces_detected:
                face = gray[y:y + h, x:x + w]
                faces.append(face)
                labels.append(current_label)
                label_map[current_label] = filename.split('.')[0]  # Employee name

            current_label += 1

    recognizer.train(faces, np.array(labels))
    return label_map

# Load trained employees
label_map = train_face_recognizer("employee_faces")

# Log violations
def log_violation(name, screenshot_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("violations_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp, screenshot_path])

# Detect safety gear and recognize employees
def detect_safety_gear_and_faces():
    logged_names_today = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Unable to capture frame.")
            break

        results = model.predict(source=frame, save=False, conf=0.5)
        safety_gear_detected = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if label in ["helmet", "gloves", "vest"]:
                    safety_gear_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not safety_gear_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                label, confidence = recognizer.predict(face)
                name = label_map.get(label, "Unknown")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"NO GEAR: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if name != "Unknown" and name not in logged_names_today:
                    screenshot_path = f"screenshots/{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                    cv2.imwrite(screenshot_path, frame)

                    log_violation(name, screenshot_path)
                    notify_hr(name, screenshot_path)
                    logged_names_today.add(name)

        cv2.imshow("Safety Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_safety_gear_and_faces()
