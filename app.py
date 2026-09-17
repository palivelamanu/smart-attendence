
from flask import Flask, render_template, request, redirect, url_for, flash, Response
import sqlite3
from datetime import datetime
import cv2
import os
import numpy as np

app = Flask(__name__)
app.secret_key = "smart-attendance-demo"
DB = "attendance.db"
FACE_DIR = "known_faces"
os.makedirs(FACE_DIR, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_no TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        face_file TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        UNIQUE(student_id, date),
        FOREIGN KEY(student_id) REFERENCES students(id)
    )""")
    conn.commit()
    conn.close()

init_db()

# Haar cascade is included with OpenCV; no external model download is required.
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def mark_attendance(student_id):
    now = datetime.now()
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO attendance(student_id,date,time) VALUES(?,?,?)",
            (student_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
        )
        conn.commit()
        result = True
    except sqlite3.IntegrityError:
        result = False
    conn.close()
    return result

@app.route("/")
def index():
    conn = get_db()
    students = conn.execute("SELECT * FROM students ORDER BY roll_no").fetchall()
    today = datetime.now().strftime("%Y-%m-%d")
    present = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE date=?", (today,)
    ).fetchone()["c"]
    conn.close()
    return render_template("index.html", students=students, present=present)

@app.route("/register", methods=["POST"])
def register():
    roll_no = request.form["roll_no"].strip()
    name = request.form["name"].strip()
    if not roll_no or not name:
        flash("Roll number and name are required.")
        return redirect(url_for("index"))

    image = request.files.get("photo")
    if image is None or image.filename == "":
        flash("Please upload a clear face photo.")
        return redirect(url_for("index"))

    filename = f"{roll_no}.jpg"
    path = os.path.join(FACE_DIR, filename)
    image.save(path)

    # Check that a face exists in the uploaded image.
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        os.remove(path)
        flash("No face detected. Upload a clear front-facing photo.")
        return redirect(url_for("index"))

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO students(roll_no,name,face_file) VALUES(?,?,?)",
            (roll_no, name, filename)
        )
        conn.commit()
        flash("Student registered successfully.")
    except sqlite3.IntegrityError:
        flash("Roll number already exists.")
        if os.path.exists(path):
            os.remove(path)
    conn.close()
    return redirect(url_for("index"))

def load_students():
    conn = get_db()
    rows = conn.execute("SELECT * FROM students").fetchall()
    conn.close()
    return rows

def recognize_face(face_gray, known_face_gray):
    # Simple demo matcher using normalized template comparison.
    # For a production system, use a trained face-embedding model.
    face = cv2.resize(face_gray, (100, 100))
    known = cv2.resize(known_face_gray, (100, 100))
    face = cv2.equalizeHist(face)
    known = cv2.equalizeHist(known)
    score = cv2.matchTemplate(face, known, cv2.TM_CCOEFF_NORMED)[0][0]
    return float(score)

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return

    students = load_students()
    known = []
    for s in students:
        path = os.path.join(FACE_DIR, s["face_file"] or "")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces = CASCADE.detectMultiScale(img, 1.2, 5)
            if len(faces):
                x,y,w,h = faces[0]
                known.append((s, img[y:y+h, x:x+w]))

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, 1.2, 5)

        for x, y, w, h in faces:
            crop = gray[y:y+h, x:x+w]
            best = None
            best_score = -1

            for s, known_face in known:
                score = recognize_face(crop, known_face)
                if score > best_score:
                    best_score = score
                    best = s

            label = "Unknown"
            if best is not None and best_score >= 0.45:
                if mark_attendance(best["id"]):
                    label = f'{best["name"]} - PRESENT'
                else:
                    label = f'{best["name"]} - ALREADY MARKED'

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    camera.release()

@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/attendance")
def attendance():
    conn = get_db()
    rows = conn.execute("""
        SELECT a.date, a.time, s.roll_no, s.name
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        ORDER BY a.date DESC, a.time DESC
    """).fetchall()
    conn.close()
    return render_template("attendance.html", rows=rows)

if __name__ == "__main__":
    app.run(debug=True)
