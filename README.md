SignDetect AI 🤟
Real-Time Sign Language Detection using Python and Machine Learning

Overview:
SignDetect AI is a real-time hand gesture and sign language detection system built using Python, computer vision, and machine learning. The application captures live video from a webcam, detects hand landmarks using MediaPipe, and classifies predefined static hand gestures using a trained Random Forest model. The project demonstrates a full end-to-end machine learning pipeline, from data collection and feature extraction to model training and real-time deployment via a Flask web application.

Features:
Real-time hand gesture recognition using a webcam, MediaPipe Hands for accurate hand landmark detection (21 key points), Random Forest classifier for gesture classification, Scale and position invariant feature normalisation, Live video streaming with prediction overlay, REST API endpoint to fetch current predictions, Modular and easy-to-understand project structure

Supported Gestures:
The model is trained on 14 static hand signs, including Common signs like HELLO, YES, NO, and Basic alphabet and numbers (A, B, 1, 2, etc.).

Tech Stack: Python, OpenCV, MediaPipe, Scikit-learn, Flask, NumPy

How It Works:
Data Collection:-
Images are captured using a webcam for each gesture class.
Feature Extraction:-
MediaPipe detects 21 hand landmarks per frame.
These landmarks are normalised to form a 42-dimensional feature vector.
Model Training:-
A Random Forest classifier is trained on the extracted features.
Real-Time Inference:-
The trained model predicts gestures from live video frames.
Web Deployment:-
Flask streams live video and provides prediction results via API.

Limitations:
Supports only static gestures (no motion-based signs)
Limited to 14 predefined classes
Single-hand detection only
Authentication system is currently a placeholder

Future Improvements:
Support for dynamic and continuous sign language using LSTM/RNNs
Larger and more diverse gesture vocabulary
Two-hand gesture recognition
Improved UI and real authentication
Deployment on mobile or edge devices

Acknowledgements:
MediaPipe Hands by Google
OpenCV community
Scikit-learn documentation

Author:
Prajwal Shivashimpar



https://github.com/user-attachments/assets/5b6a41f4-5ab8-42e4-8416-818df2fd7575

