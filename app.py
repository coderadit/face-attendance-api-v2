"""
AI Face Attendance API
-----------------------
FastAPI backend using InsightFace's small "buffalo_sc" model pack (chosen
because Render's free tier only has 512MB RAM) plus Postgres (Neon free
tier) for persistence, so enrolled faces and attendance logs survive
Render restarts/redeploys.

If the DATABASE_URL environment variable is NOT set, the app falls back
to in-memory storage automatically (handy while you're first testing
locally) — but on Render you should set DATABASE_URL so data persists.

Endpoints:
  GET  /               -> health/info
  GET  /health          -> model + database status
  POST /register        -> enroll a face (name required, student_id optional/auto-generated, file)
  POST /recognize        -> recognize a face; marks attendance if matched
  GET  /students         -> list enrolled students
  GET  /attendance        -> view attendance log (JSON), optional ?date=YYYY-MM-DD
  GET  /attendance/csv     -> download attendance log as CSV, optional ?date=YYYY-MM-DD
"""

import csv
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import psycopg2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

app = FastAPI(title="AI Face Attendance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")  # set this in Render's dashboard (Neon connection string)
IST = timezone(timedelta(hours=5, minutes=30))
MATCH_THRESHOLD = 0.45   # cosine similarity threshold for a match (buffalo_sc)
MAX_DIM = 640            # resize captured images to this max dimension

# In-memory fallback / cache. STUDENTS is always kept as a warm cache (loaded
# from Postgres at startup and updated on every /register) so face matching
# never needs a DB round trip. ATTENDANCE is only used when DATABASE_URL is
# not set at all (pure in-memory mode).
STUDENTS: dict[str, dict] = {}
ATTENDANCE: dict[str, dict[str, dict]] = {}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_conn():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    if not DATABASE_URL:
        return
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id SERIAL PRIMARY KEY,
                    date TEXT NOT NULL,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    time TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    UNIQUE(date, student_id)
                );
                """
            )
        conn.commit()
    finally:
        conn.close()


def load_students_cache():
    """Populate STUDENTS from Postgres at startup so recognition is DB-free."""
    global STUDENTS
    if not DATABASE_URL:
        return
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT student_id, name, embedding FROM students;")
            rows = cur.fetchall()
        STUDENTS = {
            sid: {"name": name, "embedding": np.array(json.loads(emb_json), dtype=np.float32)}
            for sid, name, emb_json in rows
        }
    finally:
        conn.close()


def save_student(student_id: str, name: str, embedding: np.ndarray):
    """Persist a student to Postgres (if configured) and update the cache."""
    STUDENTS[student_id] = {"name": name, "embedding": embedding}
    if not DATABASE_URL:
        return
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO students (student_id, name, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (student_id) DO UPDATE
                SET name = EXCLUDED.name, embedding = EXCLUDED.embedding;
                """,
                (student_id, name, json.dumps(embedding.tolist())),
            )
        conn.commit()
    finally:
        conn.close()


def mark_attendance(student_id: str, confidence: float):
    """Returns (already_marked: bool, date_str: str, time_str: str)."""
    date_str = today_ist()
    time_str = now_ist_time()

    if not DATABASE_URL:
        day_record = ATTENDANCE.setdefault(date_str, {})
        if student_id in day_record:
            return True, date_str, day_record[student_id]["time"]
        day_record[student_id] = {"time": time_str, "confidence": round(confidence, 2)}
        return False, date_str, time_str

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT time FROM attendance WHERE date=%s AND student_id=%s;",
                (date_str, student_id),
            )
            row = cur.fetchone()
            if row:
                return True, date_str, row[0]
            cur.execute(
                "INSERT INTO attendance (date, student_id, time, confidence) VALUES (%s,%s,%s,%s);",
                (date_str, student_id, time_str, round(confidence, 2)),
            )
        conn.commit()
        return False, date_str, time_str
    finally:
        conn.close()


def _attendance_rows(date_filter: str | None):
    if DATABASE_URL:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                if date_filter:
                    cur.execute(
                        """
                        SELECT a.date, a.student_id, s.name, a.time, a.confidence
                        FROM attendance a JOIN students s ON s.student_id = a.student_id
                        WHERE a.date = %s
                        ORDER BY a.date DESC, a.time DESC;
                        """,
                        (date_filter,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT a.date, a.student_id, s.name, a.time, a.confidence
                        FROM attendance a JOIN students s ON s.student_id = a.student_id
                        ORDER BY a.date DESC, a.time DESC;
                        """
                    )
                rows = cur.fetchall()
            return [
                {"date": d, "student_id": sid, "name": name, "time": t, "confidence": conf, "status": "Present"}
                for d, sid, name, t, conf in rows
            ]
        finally:
            conn.close()

    # in-memory fallback
    rows = []
    dates = [date_filter] if date_filter else sorted(ATTENDANCE.keys())
    for d in dates:
        for sid, info in ATTENDANCE.get(d, {}).items():
            rows.append(
                {
                    "date": d,
                    "student_id": sid,
                    "name": STUDENTS.get(sid, {}).get("name", "Unknown"),
                    "time": info["time"],
                    "confidence": info["confidence"],
                    "status": "Present",
                }
            )
    rows.sort(key=lambda r: (r["date"], r["time"]), reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Face model (lazy-loaded)
# ---------------------------------------------------------------------------
_face_app = None


def get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis

        _face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=-1, det_size=(320, 320))
    return _face_app


def load_image(raw_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((MAX_DIM, MAX_DIM))
    return np.array(img)[:, :, ::-1].copy()  # RGB -> BGR


def today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def now_ist_time() -> str:
    return datetime.now(IST).strftime("%H:%M:%S")


def slugify(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return base or "student"


def generate_student_id(name: str) -> str:
    candidate = f"{slugify(name)}-{uuid.uuid4().hex[:6]}"
    while candidate in STUDENTS:
        candidate = f"{slugify(name)}-{uuid.uuid4().hex[:6]}"
    return candidate


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    init_db()
    load_students_cache()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "AI Face Attendance API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "InsightFace (buffalo_sc)",
        "database": "connected" if DATABASE_URL else "not configured (using in-memory storage)",
        "enrolled_students": len(STUDENTS),
    }


@app.post("/register")
async def register(
    name: str = Form(...),
    student_id: str | None = Form(None),
    file: UploadFile = File(...),
):
    raw = await file.read()
    try:
        img = load_image(raw)
    except Exception:
        return {"success": False, "message": "Could not read image file"}

    fa = get_face_app()
    faces = fa.get(img)

    if len(faces) == 0:
        return {"success": False, "message": "No face detected"}
    if len(faces) > 1:
        return {"success": False, "message": "Multiple faces detected. Please use a photo with only one person"}

    embedding = faces[0].normed_embedding.astype(np.float32)

    final_id = student_id.strip() if student_id and student_id.strip() else generate_student_id(name)
    save_student(final_id, name, embedding)

    # The registration photo IS this person arriving right now, so mark attendance too.
    already_marked, date_str, time_str = mark_attendance(final_id, confidence=1.0)

    return {
        "success": True,
        "matched": True,
        "already_marked": already_marked,
        "student_id": final_id,
        "name": name,
        "confidence": 1.0,
        "date": date_str,
        "time": time_str,
        "status": "Present",
        "message": "Registered and attendance marked" if not already_marked else "Already marked present today",
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        img = load_image(raw)
    except Exception:
        return {"success": False, "matched": False, "message": "Could not read image file"}

    fa = get_face_app()
    faces = fa.get(img)

    if len(faces) == 0:
        return {"success": True, "matched": False, "message": "Face not recognized"}
    if len(faces) > 1:
        return {"success": True, "matched": False, "message": "Multiple faces detected"}

    if not STUDENTS:
        return {"success": True, "matched": False, "message": "Face not recognized"}

    query_embedding = faces[0].normed_embedding.astype(np.float32)

    best_id, best_score = None, -1.0
    for sid, record in STUDENTS.items():
        score = float(np.dot(query_embedding, record["embedding"]))
        if score > best_score:
            best_score, best_id = score, sid

    if best_score < MATCH_THRESHOLD:
        return {"success": True, "matched": False, "message": "Face not recognized"}

    name = STUDENTS[best_id]["name"]
    already_marked, date_str, time_str = mark_attendance(best_id, best_score)

    return {
        "success": True,
        "matched": True,
        "already_marked": already_marked,
        "student_id": best_id,
        "name": name,
        "confidence": round(best_score, 2),
        "date": date_str,
        "time": time_str,
        "status": "Present",
        "message": "Attendance already marked for today" if already_marked else "Attendance marked successfully",
    }


@app.get("/students")
def list_students():
    return {
        "success": True,
        "count": len(STUDENTS),
        "students": [{"student_id": sid, "name": rec["name"]} for sid, rec in STUDENTS.items()],
    }


@app.get("/attendance")
def get_attendance(date: str | None = None):
    rows = _attendance_rows(date)
    return {"success": True, "date_filter": date, "count": len(rows), "records": rows}


@app.get("/attendance/csv")
def get_attendance_csv(date: str | None = None):
    rows = _attendance_rows(date)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["date", "student_id", "name", "time", "confidence", "status"])
    writer.writeheader()
    writer.writerows(rows)

    filename = f"attendance_{date}.csv" if date else "attendance_all.csv"
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
