# main.py
import cv2
import base64
import requests
import os
import datetime # Added for timestamping
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
# IMPORTANT: Set your API keys as environment variables for security
# You can get your keys from:
# Roboflow: https://app.roboflow.com/settings/keys
# Google AI Studio: https://aistudio.google.com/app/api_keys

# CORRECTED: This now looks for environment variables named "ROBOFLOW_API_KEY" and "GEMINI_API_KEY"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Roboflow Model Details
ROBOFLOW_MODEL_ID = "license-plate-recognition-rxg4e/11"
ROBOFLOW_API_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}"

# Gemini API Details - CORRECTED to use an updated model name
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# --- ERROR HANDLING for API Keys ---
if not ROBOFLOW_API_KEY or not GEMINI_API_KEY:
    print("🚨 ERROR: API key not found.")
    print("Please set ROBOFLOW_API_KEY and GEMINI_API_KEY as environment variables before running the script.")
    exit()

def get_roboflow_predictions(frame):
    """
    Sends a frame to the Roboflow API for license plate detection.
    Args:
        frame: A numpy array representing the image frame from the webcam.
    Returns:
        A list of prediction dictionaries from the Roboflow API.
    """
    # Convert the frame to a JPEG image in memory
    _, buffer = cv2.imencode(".jpg", frame)
    # Base64 encode the image data
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Prepare headers for the API request
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    # Send the request to the Roboflow API
    response = requests.post(ROBOFLOW_API_URL, data=encoded_image, headers=headers)

    # Check for a successful response
    if response.status_code == 200:
        return response.json().get('predictions', [])
    else:
        print(f"Roboflow API Error: {response.status_code} - {response.text}")
        return []

def extract_text_with_gemini(image_bytes):
    """
    Sends a cropped image of a license plate to the Gemini API for text extraction (OCR).
    Args:
        image_bytes: The image data of the cropped license plate in bytes.
    Returns:
        The extracted text string or None if extraction fails.
    """
    # Base64 encode the image data
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # Prepare the payload for the Gemini API
    payload = {
        "contents": [{
            "parts": [
                {"text": "Extract the text from this license plate. Only return the license plate number as plain text."},
                {"inline_data": {"mime_type": "image/jpeg", "data": encoded_image}}
            ]
        }]
    }
    
    # Prepare headers for the API request
    headers = {"Content-Type": "application/json"}
    
    # Send the request to the Gemini API
    response = requests.post(GEMINI_API_URL, json=payload, headers=headers)

    # Check for a successful response and valid data
    if response.status_code == 200:
        result = response.json()
        try:
            # Navigate through the JSON response to get the text
            text = result['candidates'][0]['content']['parts'][0]['text']
            return text.strip()
        except (KeyError, IndexError):
            print("Gemini API Error: Could not parse the response.")
            return None
    else:
        print(f"Gemini API Error: {response.status_code} - {response.text}")
        return None

def main():
    """
    Main function to run the webcam license plate recognition application.
    """
    # Initialize webcam capture - UPDATED to use camera index 1
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam with index 1. Check if the camera is connected.")
        return

    print("🚀 Webcam opened successfully. Press 'q' to quit.")
    
    # --- NEW: Open a log file to save detected plates ---
    log_file_path = "detected_plates.log"
    print(f"📝 Logging detected plates to {log_file_path}")
    log_file = open(log_file_path, "a") # "a" for append mode

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Get predictions from Roboflow
        predictions = get_roboflow_predictions(frame)

        # Process each prediction
        for pred in predictions:
            # Extract bounding box coordinates
            x, y, width, height = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            
            # Calculate top-left and bottom-right corners
            x1, y1 = int(x - width / 2), int(y - height / 2)
            x2, y2 = int(x + width / 2), int(y + height / 2)
            
            # Ensure coordinates are within the frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Crop the license plate from the frame
            plate_img = frame[y1:y2, x1:x2]

            if plate_img.size > 0:
                # Convert cropped image to bytes for the Gemini API
                _, buffer = cv2.imencode('.jpg', plate_img)
                plate_bytes = buffer.tobytes()

                # Extract text from the license plate
                plate_text = extract_text_with_gemini(plate_bytes)
                
                if plate_text:
                    print(f"✅ Detected Plate Text: {plate_text}")

                    # --- NEW: Save the extracted text to the file ---
                    timestamp = datetime.datetime.now().strftime("%Y-m-d %H:%M:%S")
                    log_entry = f"{timestamp} - {plate_text}\n"
                    log_file.write(log_entry)
                    log_file.flush() # Ensure it's written immediately

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put the extracted text on the frame
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('License Plate Recognition - Press "q" to quit', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    
    # --- NEW: Close the log file ---
    log_file.close()
    print("👋 Application closed. Log file saved.")

if __name__ == "__main__":
    main()
