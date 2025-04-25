import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
from sqlalchemy import create_engine, Column, Integer, String, DateTime, and_
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
from scipy.spatial.distance import cosine
import os

app = Flask(__name__)
socketio = SocketIO(app)

DATABASE_URL = "sqlite:///attendance.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Database Models
class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    face_vector = Column(String, nullable=False)

class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)

# Load ResNet50 model
resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet50.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(tensor)
    return features.numpy().flatten()

def is_match(vector1, vector2, threshold=0.4):
    similarity = 1 - cosine(vector1, vector2)
    return similarity > (1 - threshold), similarity

# Load OpenCV face detector
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PROTOTXT_PATH = os.path.join(MODEL_DIR, 'deploy.prototxt')
MODEL_PATH = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array(
                            [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        (x1, y1, x2, y2) = box.astype('int')
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                        face = frame[y1:y2, x1:x2]
                        if face.size > 0:
                            face_vector = extract_features(face)

                            students = session.query(Student).all()
                            for student in students:
                                db_vector = np.array([float(x) for x in student.face_vector.strip("[]").split(',')])
                                match, similarity = is_match(face_vector, db_vector)
                                if match:
                                    today = datetime.datetime.utcnow().date()
                                    start_time = datetime.datetime.combine(today, datetime.time.min)
                                    end_time = datetime.datetime.combine(today, datetime.time.max)

                                    existing_record = session.query(Attendance).filter(
                                        and_(
                                            Attendance.student_id == student.id,
                                            Attendance.timestamp >= start_time,
                                            Attendance.timestamp <= end_time
                                        )
                                    ).first()

                                    if not existing_record:
                                        attendance = Attendance(student_id=student.id)
                                        session.add(attendance)
                                        session.commit()
                                        socketio.emit('attendance_update', {
                                            'student_name': student.name,
                                            'timestamp': str(datetime.datetime.utcnow())
                                        })
                                    break

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        finally:
            cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register_student', methods=['POST'])
def register_student():
    name = request.form.get('name')
    file = request.files['image']
    if not name or not file:
        return jsonify({"status": "error", "message": "Name and image are required."})

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    face_vector = extract_features(image)
    student = Student(name=name, face_vector=str(face_vector.tolist()))
    session.add(student)
    session.commit()
    return jsonify({"status": "success", "message": f"Student '{name}' registered successfully."})

if __name__ == '__main__':
    socketio.run(app, debug=True)
