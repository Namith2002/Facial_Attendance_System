import sqlite3
import hashlib
from datetime import datetime
import os

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Face encodings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Attendance records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'present',
            confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    """Verify user credentials"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute('''
        SELECT id, username, full_name, email FROM users 
        WHERE username = ? AND password_hash = ?
    ''', (username, password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {
            'id': user[0],
            'username': user[1],
            'full_name': user[2],
            'email': user[3]
        }
    return None

def create_user(username, password, full_name, email):
    """Create a new user"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, full_name, email)
            VALUES (?, ?, ?, ?)
        ''', (username, password_hash, full_name, email))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def save_attendance(user_id, confidence):
    """Save attendance record"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO attendance (user_id, confidence)
        VALUES (?, ?)
    ''', (user_id, confidence))
    
    conn.commit()
    conn.close()

def get_user_attendance(user_id, date=None):
    """Get attendance records for a user"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    if date:
        cursor.execute('''
            SELECT timestamp, status, confidence FROM attendance
            WHERE user_id = ? AND DATE(timestamp) = ?
            ORDER BY timestamp DESC
        ''', (user_id, date))
    else:
        cursor.execute('''
            SELECT timestamp, status, confidence FROM attendance
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
    
    records = cursor.fetchall()
    conn.close()
    return records

def get_today_attendance():
    """Get today's attendance for all users"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT u.full_name, a.timestamp, a.confidence
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE DATE(a.timestamp) = DATE('now')
        ORDER BY a.timestamp DESC
    ''', )
    
    records = cursor.fetchall()
    conn.close()
    return records