import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from threading import Thread, Lock
import time

# --- 1. MODEL & CONFIGURATION SETUP (Based on your uploaded files) ---

# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: model.p not found. Run train_classifier.py first.")
    model = None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, 
                       min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Label mapping (based on inference_classifier.py snippet)
labels_dict = {0: 'HELLO', 1: 'YES', 2: 'NO', 3: 'THANKS', 4: 'SORRY', 5: 'PLEASE', 6: 'A', 7: 'B', 8: 'C', 9: 'D', 10: '1', 11: '2', 12: '3', 13: '4'}


# --- 2. GLOBAL STATE FOR REAL-TIME PREDICTION ---

# Variables to hold the current frame and prediction text
global_frame = None
global_prediction = "Waiting to detect a sign..."
frame_lock = Lock()


# --- 3. VIDEO PROCESSING AND STREAMING (Modified inference_classifier.py logic) ---

class VideoCamera(object):
    """
    Handles video capture and real-time inference logic.
    """
    def __init__(self):
        # Use cv2.VideoCapture(0) to access the webcam
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Error: Could not open camera.")
            exit()
        
        # Start the thread to read frames from the camera
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def __del__(self):
        # Release the camera when the object is destroyed
        self.video.release()

    def update(self):
        """Continuously captures frames and runs inference."""
        global global_frame, global_prediction
        while True:
            ret, frame = self.video.read()
            if not ret or frame is None:
                # Break the loop if camera read fails
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # --- Inference Logic (Identical to inference_classifier.py) ---
            
            data_aux = []
            x_ = []
            y_ = []
            current_prediction = "No Hand Detected"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    # 1. Draw Landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # 2. Extract and Normalize features
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        # Append normalized coordinates (x - min(x), y - min(y))
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                
                # Check if features are ready (must be 42 features)
                if len(data_aux) == 42 and model is not None:
                    # 3. Predict the sign
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(int(prediction[0]), 'UNKNOWN')
                    current_prediction = predicted_character

                    # 4. Draw bounding box and text
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10 # Adjusted for better visibility
                    y2 = int(max(y_) * H) + 10 # Adjusted for better visibility
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    
            # Update global state for streaming
            with frame_lock:
                global_frame = frame
                global_prediction = current_prediction
            
    def get_frame(self):
        """Converts the processed frame into JPEG format for web streaming."""
        global global_frame
        with frame_lock:
            if global_frame is None:
                # Return a placeholder image if the camera hasn't started or failed
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, 'Camera Loading...', (100, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', blank_frame)
            else:
                ret, jpeg = cv2.imencode('.jpg', global_frame)
            
        return jpeg.tobytes()

# Initialize the camera object globally
cam = VideoCamera()

def gen(camera):
    """Generator function that yields JPEG frames for the video stream."""
    while True:
        frame = camera.get_frame()
        # MJPEG streaming format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- 4. FLASK APPLICATION AND ROUTES ---

app = Flask(__name__, template_folder='.') # Use current directory for HTML files

# --- Navigation Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how_it_works.html')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/demo.html')
def demo():
    return render_template('demo.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

# --- Authentication Routes (STUBS) ---

@app.route('/signup.html')
def signup_page():
    return render_template('signup.html')

@app.route('/login.html')
def login_page():
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup_submit():
    # Placeholder for actual authentication logic
    username = request.form.get('username')
    # In a real app, you would hash the password and save the user to a database.
    print(f"STUB: User {username} attempting to sign up.")
    # Redirect to login page after successful signup (simulated)
    return redirect(url_for('login_page')) 

@app.route('/login', methods=['POST'])
def login_submit():
    # Placeholder for actual authentication logic
    username_email = request.form.get('username_email')
    # In a real app, you would verify the credentials against the database.
    print(f"STUB: User {username_email} attempting to log in.")
    # Redirect to demo page after successful login (simulated)
    return redirect(url_for('demo')) 

# --- Inference API Routes ---

@app.route('/video_feed')
def video_feed():
    """Route to stream the processed video frames to the browser."""
    # The 'gen' function generates the frames, which are served as an MJPEG stream.
    return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    """Route to return the current predicted sign text as JSON."""
    global global_prediction
    # Return the current prediction text
    return jsonify({'prediction': global_prediction})


if __name__ == '__main__':
    # Ensure the CSS file is accessible
    # (If your CSS is in a 'static' folder, adjust Flask app initialization)
    if not os.path.exists('style.css'):
         print("Warning: style.css not found in the same directory. Ensure files are placed correctly.")
         
    # Run the Flask app on the default port
    print("\n\n--- SERVER STARTING ---")
    print("Open your browser to: http://127.0.0.1:5000/")
    print("Press Ctrl+C to stop.")
    app.run(host='0.0.0.0', threaded=True) # threaded=True allows video streaming and routing to work concurrently