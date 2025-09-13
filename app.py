from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import cv2
import os
from werkzeug.utils import secure_filename
import base64
import numpy as np
from datetime import datetime, date
import json

from database import init_db, verify_user, create_user, save_attendance, get_user_attendance, get_today_attendance
from face_utils import save_face_encoding, recognize_faces_in_frame, has_face_registered, allowed_file

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
init_db()

# Global camera object
camera = None

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = verify_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    full_name = request.form['full_name']
    email = request.form['email']
    
    user_id = create_user(username, password, full_name, email)
    if user_id:
        return jsonify({'success': True, 'message': 'User created successfully'})
    else:
        return jsonify({'success': False, 'message': 'Username or email already exists'})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Check if user has registered face
    face_registered = has_face_registered(user_id)
    
    # Get user's attendance records
    attendance_records = get_user_attendance(user_id)
    
    # Get today's attendance
    today_attendance = get_today_attendance()
    
    return render_template('dashboard.html', 
                         face_registered=face_registered,
                         attendance_records=attendance_records,
                         today_attendance=today_attendance)

@app.route('/register_face', methods=['GET', 'POST'])
def register_face():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'face_image' not in request.files:
            return jsonify({'success': False, 'message': 'No image uploaded'})
        
        file = request.files['face_image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No image selected'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"user_{session['user_id']}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            success, message = save_face_encoding(session['user_id'], file_path)
            return jsonify({'success': success, 'message': message})
        
        return jsonify({'success': False, 'message': 'Invalid file format'})
    
    return render_template('register_face.html')

@app.route('/attendance')
def attendance():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if not has_face_registered(session['user_id']):
        return redirect(url_for('register_face'))
    
    return render_template('attendance.html')

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Recognize faces
        face_locations, face_data = recognize_faces_in_frame(frame)
        
        # Draw rectangles and names
        for (top, right, bottom, left), (name, user_data) in zip(face_locations, face_data):
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    global camera
    if camera is None:
        return jsonify({'success': False, 'message': 'Camera not initialized'})
    
    # Capture current frame
    success, frame = camera.read()
    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture frame'})
    
    # Recognize faces
    face_locations, face_data = recognize_faces_in_frame(frame)
    
    user_id = session['user_id']
    user_recognized = False
    confidence = 0
    
    for (name, user_data) in face_data:
        if user_data[0] == user_id:  # Check if current user is recognized
            user_recognized = True
            confidence = user_data[1]
            break
    
    if user_recognized:
        save_attendance(user_id, confidence)
        return jsonify({'success': True, 'message': f'Attendance marked successfully! (Confidence: {confidence:.2f})'})
    else:
        return jsonify({'success': False, 'message': 'Face not recognized. Please ensure your face is clearly visible.'})

@app.route('/mark_attendance_with_photo', methods=['POST'])
def mark_attendance_with_photo():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    if 'face_image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    file = request.files['face_image']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'})
    
    if file and allowed_file(file.filename):
        # Save the file temporarily
        temp_filename = secure_filename(f"temp_{session['user_id']}_{file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)
        
        # Load the image and recognize faces
        frame = cv2.imread(temp_path)
        if frame is None:
            os.remove(temp_path)  # Clean up
            return jsonify({'success': False, 'message': 'Failed to process image'})
        
        # Recognize faces
        face_locations, face_data = recognize_faces_in_frame(frame)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        user_id = session['user_id']
        user_recognized = False
        confidence = 0
        
        for (name, user_data) in face_data:
            if user_data[0] == user_id:  # Check if current user is recognized
                user_recognized = True
                confidence = user_data[1]
                break
        
        if user_recognized:
            save_attendance(user_id, confidence)
            return jsonify({'success': True, 'message': f'Attendance marked successfully! (Confidence: {confidence:.2f})'})
        else:
            return jsonify({'success': False, 'message': 'Face not recognized. Please ensure your face is clearly visible.'})
    
    return jsonify({'success': False, 'message': 'Invalid file format'})

@app.route('/get_attendance_stats')
def get_attendance_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'})
    
    user_id = session['user_id']
    
    # Get attendance for last 7 days
    records = get_user_attendance(user_id)
    
    # Process data for charts
    daily_attendance = {}
    for record in records:
        date_str = record[0].split()[0]  # Get date part
        if date_str not in daily_attendance:
            daily_attendance[date_str] = 0
        daily_attendance[date_str] += 1
    
    return jsonify({
        'daily_attendance': daily_attendance,
        'total_records': len(records)
    })

@app.route('/logout')
def logout():
    global camera
    if camera:
        camera.release()
        camera = None
    
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)