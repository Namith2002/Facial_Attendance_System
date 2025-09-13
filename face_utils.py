import face_recognition
import cv2
import numpy as np
import sqlite3
import pickle
import os
from werkzeug.utils import secure_filename

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_face_encoding(user_id, image_path):
    """Extract and save face encoding from image"""
    try:
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            return False, "No face detected in the image"
        
        if len(face_encodings) > 1:
            return False, "Multiple faces detected. Please use an image with only one face"
        
        # Get the first (and only) face encoding
        face_encoding = face_encodings[0]
        
        # Save to database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Convert numpy array to blob
        encoding_blob = pickle.dumps(face_encoding)
        
        cursor.execute('''
            INSERT INTO face_encodings (user_id, encoding, image_path)
            VALUES (?, ?, ?)
        ''', (user_id, encoding_blob, image_path))
        
        conn.commit()
        conn.close()
        
        return True, "Face registered successfully"
        
    except Exception as e:
        return False, f"Error processing image: {str(e)}"

def get_all_face_encodings():
    """Get all face encodings from database"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT fe.user_id, fe.encoding, u.full_name
        FROM face_encodings fe
        JOIN users u ON fe.user_id = u.id
    ''')
    
    records = cursor.fetchall()
    conn.close()
    
    known_encodings = []
    known_user_ids = []
    known_names = []
    
    for record in records:
        user_id, encoding_blob, name = record
        encoding = pickle.loads(encoding_blob)
        known_encodings.append(encoding)
        known_user_ids.append(user_id)
        known_names.append(name)
    
    return known_encodings, known_user_ids, known_names

def recognize_faces_in_frame(frame):
    """Recognize faces in a video frame"""
    # Get known faces
    known_encodings, known_user_ids, known_names = get_all_face_encodings()
    
    if not known_encodings:
        return [], []
    
    try:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB
        
        # Ensure the frame is in the correct format
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)
        
        # Find faces in current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Only proceed if faces are found
        if not face_locations:
            return [], []
            
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
        face_names = []
        face_user_ids = []
        
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
                user_id = known_user_ids[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                
                face_names.append(f"{name} ({confidence:.2f})")
                face_user_ids.append((user_id, confidence))
            else:
                face_names.append("Unknown")
                face_user_ids.append((None, 0))
        
        # Scale back face locations
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]
        
        return face_locations, list(zip(face_names, face_user_ids))
        
    except Exception as e:
        print(f"Error in face recognition: {str(e)}")
        return [], []

def has_face_registered(user_id):
    """Check if user has registered face"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM face_encodings WHERE user_id = ?', (user_id,))
    count = cursor.fetchone()[0]
    
    conn.close()
    return count > 0