import cv2
import os
import csv
import smtplib
import mimetypes
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import shutil

model = YOLO("./Safety-Detection-YOLOv8/ppe.pt")  

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

GMAIL_USER = "221701038@rajalakshmi.edu.in"
GMAIL_PASSWORD = "221701038"  
HR_EMAIL = "monishdhanush75@gmail.com"


FACE_DETECTION_SCALE = 1.2  
FACE_DETECTION_NEIGHBORS = 6  
MIN_FACE_SIZE = (50, 50)
RECOGNITION_CONFIDENCE_THRESHOLD = 100  

REQUIRED_SAFETY_GEAR = ["helmet", "vest", "mask"]

def send_email(to_email, subject, body, image_path):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    
    # Add text content
    text_part = MIMEText(body)
    msg.attach(text_part)
    
    # Add the annotated image with detection boxes
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        image = MIMEImage(img_data)
        image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        msg.attach(image)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False

# Function to notify HR when safety violation occurs
def notify_hr(name, missing_gear, annotated_image_path):
    subject = f"SAFETY ALERT: {name} - Missing Safety Gear"
    body = f"""SAFETY VIOLATION DETECTED

Employee: {name}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Violation: Missing required safety gear: {', '.join(missing_gear)}

This is an automated alert from the Safety Detection System.
The attached image shows the detection with highlighted areas.
"""
    return send_email(HR_EMAIL, subject, body, annotated_image_path)

# Extract faces from images for training
def extract_faces_for_training(input_dir, output_dir):
    """
    Processes images in input_dir and extracts detected faces to output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each image in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_path}")
                continue
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization to improve contrast
            gray = cv2.equalizeHist(gray)
            
            # Detect faces - try with multiple scale factors
            all_faces = []
            for scale in [1.05, 1.1, 1.15, 1.2]:
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale, 
                    minNeighbors=FACE_DETECTION_NEIGHBORS,
                    minSize=MIN_FACE_SIZE
                )
                if len(faces) > 0:
                    all_faces.extend(faces)
                    break  # Use the first successful scale factor
            
            # If faces found, save them
            face_count = 0
            for i, (x, y, w, h) in enumerate(all_faces):
                face_img = image[y:y+h, x:x+w]
                if face_img.size > 0:  # Ensure the face extraction was successful
                    output_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, face_img)
                    face_count += 1
            
            if face_count > 0:
                print(f"Extracted {face_count} faces from {filename}")
            else:
                print(f"No faces found in {filename} - try different lighting or angles")
    
    return face_count > 0

# Data augmentation for training
def augment_image(image):
    """Generate variations of the input image for better training"""
    augmented_images = [image]
    
    # Brightness variations
    for alpha in [0.8, 0.9, 1.1, 1.2]:
        bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented_images.append(bright)
    
    # Small rotations
    for angle in [-15, -10, -5, 5, 10, 15]:  # Added more angles
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)
    
    # Horizontal flip (important for face recognition)
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # Add slight blur for robustness
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmented_images.append(blurred)
    
    # Add slight noise
    noise = np.copy(image)
    cv2.randn(noise, 0, 15)
    noisy = cv2.add(image, noise)
    augmented_images.append(noisy)
    
    return augmented_images

# Prepare the training data with face extraction and augmentation
def prepare_training_data(face_folder_path):
    """Prepare training data from employee photos with face extraction and augmentation"""
    print("Preparing training data...")
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    
    # Create a directory for processed faces
    processed_dir = os.path.join(os.path.dirname(face_folder_path), "processed_faces")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process each person's directory
    for person_name in os.listdir(face_folder_path):
        person_path = os.path.join(face_folder_path, person_name)
        if os.path.isdir(person_path):
            print(f"Processing {person_name}'s images...")
            
            # Create output directory for this person's extracted faces
            person_output_dir = os.path.join(processed_dir, person_name)
            os.makedirs(person_output_dir, exist_ok=True)
            
            # Extract faces from this person's images
            success = extract_faces_for_training(person_path, person_output_dir)
            
            if not success:
                print(f"⚠️ Warning: Could not extract any faces for {person_name}")
                continue
                
            # Now use the extracted faces for training
            person_images_count = 0
            for filename in os.listdir(person_output_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_output_dir, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Standardize size for consistent training
                    resized = cv2.resize(gray, (100, 100))
                    
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(resized)
                    
                    # Add original face
                    faces.append(enhanced)
                    labels.append(current_label)
                    person_images_count += 1
                    
                    # Add augmented variations
                    augmented_faces = augment_image(enhanced)
                    for aug_face in augmented_faces[1:]:  # Skip the first one (original)
                        faces.append(aug_face)
                        labels.append(current_label)
                        person_images_count += 1
            
            if person_images_count > 0:
                print(f"Added {person_images_count} face samples for {person_name}")
                label_map[current_label] = person_name
                current_label += 1
            else:
                print(f"⚠️ Warning: No usable face images for {person_name}")
    
    return faces, labels, label_map

# Train face recognizer
def train_face_recognizer():
    """Train the face recognizer using employee images"""
    print("Starting face recognition training...")
    
    # Check if employee_faces directory exists
    if not os.path.exists("employee_faces"):
        print("❌ Error: 'employee_faces' directory not found.")
        print("Please create this directory and add subfolders for each employee with their photos.")
        return None
    
    # Prepare data with face extraction and augmentation
    faces, labels, label_map = prepare_training_data("employee_faces")
    
    if not faces:
        print("❌ Error: No faces detected in the training images!")
        return None
    
    print(f"Training recognizer on {len(faces)} face samples for {len(label_map)} people...")
    try:
        # Train the recognizer with optimized parameters
        recognizer.setRadius(1)  # Set radius for LBPH
        recognizer.setNeighbors(8)  # Set neighbors for LBPH
        recognizer.setGridX(8)  # More cells in X direction
        recognizer.setGridY(8)  # More cells in Y direction
        
        recognizer.train(faces, np.array(labels))
        print("Training complete!")
        
        # Save the trained model
        model_dir = "face_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "trained_face_model.yml")
        recognizer.write(model_path)
        print(f"Model saved to {model_path}")
        
        # Save the label mapping
        label_map_path = os.path.join(model_dir, "label_map.txt")
        with open(label_map_path, 'w') as f:
            for label, name in label_map.items():
                f.write(f"{label}:{name}\n")
        print(f"Label map saved to {label_map_path}")
        
        return label_map
    
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

# Load label mapping from file
def load_label_map():
    """Load the label mapping from the saved file"""
    label_map = {}
    label_map_path = os.path.join("face_models", "label_map.txt")
    
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        label_map[int(parts[0])] = parts[1]
            print(f"Loaded label map with {len(label_map)} entries")
        except Exception as e:
            print(f"Error loading label map: {e}")
    else:
        print("Label map file not found")
    
    return label_map

# Log violations
def log_violation(name, missing_gear, screenshot_path):
    """Log safety violations to a CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "violations_log.csv")
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Missing Gear", "Timestamp", "Screenshot"])
    
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, ", ".join(missing_gear), timestamp, screenshot_path])
    
    print(f"Violation logged: {name} missing {missing_gear} at {timestamp}")

# Force retrain model
def force_retrain():
    """Force retraining the face recognition model"""
    # Remove existing model files
    model_dir = "face_models"
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print("Removed existing model files for retraining")
    
    # Remove processed faces directory
    processed_dir = "processed_faces"
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
        print("Removed processed faces for retraining")
    
    # Train new model
    return train_face_recognizer()

# Detect safety gear and recognize employees
def detect_safety_gear_and_faces(force_train=False):
    """Main function to detect safety gear and recognize employees"""
    global RECOGNITION_CONFIDENCE_THRESHOLD
    print("Starting safety detection and face recognition...")
    
    # Force retraining if requested
    if force_train:
        print("Forcing model retraining...")
        label_map = force_retrain()
    else:
        # Try to load a previously trained model
        model_path = os.path.join("face_models", "trained_face_model.yml")
        if os.path.exists(model_path):
            try:
                recognizer.read(model_path)
                print(f"Loaded pre-trained face model from {model_path}")
                label_map = load_label_map()
                if not label_map:
                    print("Label map not found or empty. Retraining model...")
                    label_map = train_face_recognizer()
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Training new model instead...")
                label_map = train_face_recognizer()
        else:
            print("No pre-trained model found. Training new model...")
            label_map = train_face_recognizer()
    
    if not label_map:
        print("❌ Error: Could not train or load face recognition model.")
        return
    
    print(f"Starting camera with {len(label_map)} registered employees")
    print(f"Registered employees: {', '.join(label_map.values())}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera. Check camera connection.")
        return
    
    # Notification tracking
    notified_employees = set()  # Employees notified today
    last_notification_time = {}  # Last notification time for each employee
    notification_cooldown = 300  # 5 minutes between notifications for the same person
    
    # Create directories for outputs
    os.makedirs("screenshots", exist_ok=True)
    
    # Create a window and set it to fullscreen
    cv2.namedWindow("Safety Detection", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Unable to capture frame.")
            break

        # Make a copy for display
        display_frame = frame.copy()
        
        # Step 1: Detect safety gear using YOLO
        results = model.predict(source=frame, save=False, conf=0.5)
        
        # Track detected safety gear
        detected_gear = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if label in REQUIRED_SAFETY_GEAR and confidence > 0.5:
                    detected_gear.add(label)
                    color = (0, 255, 0)  # Green for safety gear
                else:
                    color = (0, 165, 255)  # Orange for other objects
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label} {confidence:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate missing gear
        missing_gear = [gear for gear in REQUIRED_SAFETY_GEAR if gear not in detected_gear]
        safety_compliant = len(missing_gear) == 0
        
        # Step 2: Recognize faces - try multiple processing approaches
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Detect faces with multiple scale factors
        all_faces = []
        for scale in [1.05, 1.1, 1.15, 1.2]:
            faces = face_cascade.detectMultiScale(
                enhanced_gray, 
                scaleFactor=scale, 
                minNeighbors=FACE_DETECTION_NEIGHBORS, 
                minSize=MIN_FACE_SIZE
            )
            if len(faces) > 0:
                all_faces = faces
                break
        
        current_time = datetime.now()
        violations_detected = False
        
        # Process each detected face
        for (x, y, w, h) in all_faces:
            # Extract and process the face 
            face_img = gray[y:y+h, x:x+w]
            
            # Skip if face is too small
            if face_img.shape[0] < 30 or face_img.shape[1] < 30:
                continue
                
            # Apply CLAHE for better contrast
            face_enhanced = clahe.apply(face_img)
            
            # Resize for consistent recognition
            resized_face = cv2.resize(face_enhanced, (100, 100))
            
            # Try recognition
            best_confidence = float('inf')  # Lower is better for LBPH
            best_label = -1
            
            # Try different pre-processing methods and use best result
            for process_method in [resized_face, 
                                  cv2.GaussianBlur(resized_face, (5, 5), 0),
                                  cv2.equalizeHist(resized_face)]:
                try:
                    label, confidence = recognizer.predict(process_method)
                    if confidence < best_confidence:
                        best_confidence = confidence
                        best_label = label
                except Exception as e:
                    pass
            
            # Attempt recognition with the best result
            if best_label != -1:
                # Lower values mean better match in LBPH
                if best_confidence < RECOGNITION_CONFIDENCE_THRESHOLD:
                    name = label_map.get(best_label, "Unknown")
                    
                    # Set color based on safety compliance
                    if safety_compliant:
                        box_color = (0, 255, 0)  # Green for compliant
                        status_text = f"{name} (Compliant)"
                    else:
                        box_color = (0, 0, 255)  # Red for non-compliant
                        status_text = f"NO GEAR: {name}"
                        violations_detected = True
                        
                        # Check if we should notify about this person
                        should_notify = False
                        if name in last_notification_time:
                            time_since_last = (current_time - last_notification_time[name]).total_seconds()
                            if time_since_last > notification_cooldown:
                                should_notify = True
                        else:
                            should_notify = True
                        
                        if should_notify and name not in notified_employees:
                            # Take a screenshot for the violation - with annotations
                            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                            screenshot_path = os.path.join("screenshots", f"{name}_{timestamp}.jpg")
                            
                            # Add violation details to the display frame
                            violation_text = f"Missing: {', '.join(missing_gear)}"
                            cv2.putText(display_frame, violation_text, 
                                      (10, display_frame.shape[0] - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Save the annotated frame
                            cv2.imwrite(screenshot_path, display_frame)
                            
                            # Log the violation
                            log_violation(name, missing_gear, screenshot_path)
                            
                            # Notify HR
                            success = notify_hr(name, missing_gear, screenshot_path)
                            if success:
                                notified_employees.add(name)
                                last_notification_time[name] = current_time
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(display_frame, status_text, 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    cv2.putText(display_frame, f"Conf: {best_confidence:.1f}", 
                              (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                else:
                    # Unknown person
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
                    cv2.putText(display_frame, f"Unknown ({best_confidence:.1f})", 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            else:
                # Recognition failed
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_frame, "Error", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add status information to display
        if safety_compliant:
            status_text = "SAFE: All required safety gear detected"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = f"ALERT: Missing safety gear: {', '.join(missing_gear)}"
            status_color = (0, 0, 255)  # Red
        
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show the number of registered employees
        cv2.putText(display_frame, f"Registered employees: {len(label_map)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show recognition confidence threshold
        cv2.putText(display_frame, f"Recognition threshold: {RECOGNITION_CONFIDENCE_THRESHOLD}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow("Safety Detection", display_frame)
        
        # Break the loop when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Increase confidence threshold with + key
        elif key == ord('+') or key == ord('='):
            RECOGNITION_CONFIDENCE_THRESHOLD += 10
            print(f"Recognition threshold increased to {RECOGNITION_CONFIDENCE_THRESHOLD}")
        # Decrease confidence threshold with - key
        elif key == ord('-') or key == ord('_'):
            RECOGNITION_CONFIDENCE_THRESHOLD = max(10, RECOGNITION_CONFIDENCE_THRESHOLD - 10)
            print(f"Recognition threshold decreased to {RECOGNITION_CONFIDENCE_THRESHOLD}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Safety detection stopped.")

# Main execution
if __name__ == "__main__":
    print("=== Safety Detection System ===")
    print("1. Start detection with existing model")
    print("2. Force retrain model and start detection")
    choice = input("Enter your choice (1/2): ")
    
    if choice == "2":
        print("Forcing model retraining...")
        detect_safety_gear_and_faces(force_train=True)
    else:
        print("Starting with existing model (if available)...")
        detect_safety_gear_and_faces(force_train=False)
    
    print("Press 'q' to exit the application when the camera window is active")
    print("Press '+' to increase recognition threshold")
    print("Press '-' to decrease recognition threshold")        
       