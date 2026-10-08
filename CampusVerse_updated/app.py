from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, Response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import json
import base64
import hashlib
import threading
import time

from models import db, User, StudentProfile, FacultyProfile, Announcement, Event, Note, Assignment, Attendance, AnnouncementSeen
from tts_service import TTSService
from utils.helpers import allowed_file, format_datetime, get_notifications_count
import train_faces as knn_train
import cv2
import face_recognition
import numpy as np
import pickle

# Initialize Flask app
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "database", "campusverse.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize services
tts_service = TTSService()

# Model path constant
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.clf')


# ==================== FACE RECOGNITION HELPERS ====================

def identify_faces_from_frame(rgb_frame, model_path=MODEL_PATH, distance_threshold=0.5):
    """Identify faces in a single frame and return names with locations."""
    try:
        if not os.path.exists(model_path):
            app.logger.error(f"Model not found at {model_path}")
            return []

        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

        if len(face_locations) == 0:
            return []

        face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)

        if len(face_encodings) == 0:
            return []

        closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

        return [(pred, loc) if rec else ("unknown", loc)
                for pred, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches)]
    except Exception as e:
        app.logger.error(f"Face identification error: {str(e)}")
        return []


def mark_attendance_by_face(username, course, faculty_id):
    """Mark attendance for a recognized student. Returns True if marked."""
    try:
        # Folder names may be "username_0" so try exact match first, then prefix match
        user = User.query.filter_by(username=username).first()

        if not user:
            # Try matching with underscore prefix (e.g., dataset folder "john_0")
            user = User.query.filter(User.username.like(f"{username}_%")).first()

        if not user or user.role != 'student':
            app.logger.warning(f"Student not found for username: {username}")
            return False

        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        existing = Attendance.query.filter_by(
            student_id=user.id,
            course=course,
            date=today
        ).first()

        if existing:
            return False

        attendance = Attendance(
            student_id=user.id,
            course=course,
            date=today,
            status='present',
            marked_by=faculty_id,
            marked_at=datetime.utcnow()
        )

        db.session.add(attendance)
        db.session.commit()

        # Update attendance percentage
        student_attendance = Attendance.query.filter_by(student_id=user.id).all()
        total = len(student_attendance)
        present = len([a for a in student_attendance if a.status == 'present'])
        percentage = (present / total * 100) if total > 0 else 0

        profile = StudentProfile.query.filter_by(user_id=user.id).first()
        if profile:
            profile.attendance_percentage = percentage
            db.session.commit()

        app.logger.info(f"Attendance marked for {username} in {course}")
        return True

    except Exception as e:
        app.logger.error(f"Error in mark_attendance_by_face: {str(e)}")
        db.session.rollback()
        return False


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def init_db():
    with app.app_context():
        db.create_all()

        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@campusverse.com',
                full_name='System Administrator',
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created successfully!")


def get_all_courses():
    """Get list of distinct courses from student profiles."""
    courses = db.session.query(StudentProfile.course).distinct().all()
    return sorted([c[0] for c in courses if c[0]])


# ==================== MAIN ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password) and user.is_active:
            login_user(user)
            app.logger.info(f'User {user.username} logged in successfully')

            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'faculty':
                return redirect(url_for('faculty_dashboard'))
            else:
                return redirect(url_for('student_dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        role = request.form.get('role', 'student')
        student_id = request.form.get('student_id')
        course = request.form.get('course')
        year = request.form.get('year')

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))

        user = User(
            username=username,
            email=email,
            full_name=full_name,
            role=role
        )
        user.set_password(password)
        db.session.add(user)
        db.session.flush()

        if role == 'student':
            if not student_id:
                flash('Student ID is required for student registration', 'danger')
                return redirect(url_for('register'))

            profile = StudentProfile(
                user_id=user.id,
                student_id=student_id,
                course=course,
                year=int(year) if year else 1
            )
            db.session.add(profile)
            db.session.commit()

            # Capture face samples
            cam = cv2.VideoCapture(0)
            img_counter = 0
            DIR = f"./Dataset/{username}_0"
            try:
                os.makedirs(DIR, exist_ok=True)
                print(f"Directory {DIR} created/exists")
            except Exception as e:
                print(f"Directory creation error: {e}")

            img_counter = len(os.listdir(DIR))

            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                cv2.imshow("Press SPACE to capture, ESC to finish", frame)
                k = cv2.waitKey(1)
                if k % 256 == 27:
                    break
                elif k % 256 == 32:
                    img_name = f"{DIR}/opencv_frame_{img_counter}.png"
                    cv2.imwrite(img_name, frame)
                    print(f"{img_name} written!")
                    img_counter += 1
            cam.release()
            cv2.destroyAllWindows()

            try:
                knn_train.trainer()
            except Exception as e:
                print(f"Training error: {e}")

        elif role == 'faculty':
            faculty_id = request.form.get('faculty_id')
            department = request.form.get('department')

            if not faculty_id:
                flash('Faculty ID is required for faculty registration', 'danger')
                return redirect(url_for('register'))

            profile = FacultyProfile(
                user_id=user.id,
                faculty_id=faculty_id,
                department=department
            )
            db.session.add(profile)
            db.session.commit()
        else:
            db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


# ==================== STUDENT DASHBOARD ====================

@app.route('/student')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    student_profile = StudentProfile.query.filter_by(user_id=current_user.id).first()

    announcements = Announcement.query.filter(
        Announcement.expires_at.is_(None) | (Announcement.expires_at > datetime.utcnow())
    ).order_by(Announcement.created_at.desc()).limit(10).all()

    events = Event.query.filter_by(is_active=True).order_by(Event.event_date.asc()).limit(5).all()
    notes = Note.query.order_by(Note.uploaded_at.desc()).limit(5).all()
    assignments = Assignment.query.order_by(Assignment.uploaded_at.desc()).limit(5).all()

    attendance_records = Attendance.query.filter_by(student_id=current_user.id).all()
    total_classes = len(attendance_records)
    present_classes = len([a for a in attendance_records if a.status == 'present'])
    attendance_percentage = (present_classes / total_classes * 100) if total_classes > 0 else 0

    if student_profile:
        student_profile.attendance_percentage = attendance_percentage
        db.session.commit()

    notifications_count = get_notifications_count(current_user.id)

    return render_template('student_dashboard.html',
                           student_profile=student_profile,
                           announcements=announcements,
                           events=events,
                           notes=notes,
                           assignments=assignments,
                           attendance_percentage=attendance_percentage,
                           notifications_count=notifications_count)


# ==================== FACULTY DASHBOARD ====================

@app.route('/faculty')
@login_required
def faculty_dashboard():
    if current_user.role != 'faculty':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    faculty_profile = FacultyProfile.query.filter_by(user_id=current_user.id).first()
    announcements = Announcement.query.filter_by(created_by=current_user.id).order_by(Announcement.created_at.desc()).limit(10).all()
    notes = Note.query.filter_by(uploaded_by=current_user.id).order_by(Note.uploaded_at.desc()).limit(5).all()
    assignments = Assignment.query.filter_by(uploaded_by=current_user.id).order_by(Assignment.uploaded_at.desc()).limit(5).all()

    return render_template('faculty_dashboard.html',
                           faculty_profile=faculty_profile,
                           announcements=announcements,
                           notes=notes,
                           assignments=assignments,
                           speaker_status=tts_service.get_speaker_status())


# ==================== ADMIN DASHBOARD ====================

@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    total_users = User.query.count()
    total_students = StudentProfile.query.count()
    total_faculty = FacultyProfile.query.count()
    total_announcements = Announcement.query.count()
    total_events = Event.query.count()
    total_attendance = Attendance.query.count()

    return render_template('admin_dashboard.html',
                           total_users=total_users,
                           total_students=total_students,
                           total_faculty=total_faculty,
                           total_announcements=total_announcements,
                           total_events=total_events,
                           total_attendance=total_attendance,
                           speaker_status=tts_service.get_speaker_status())


# ==================== STUDENT FEATURES ====================

@app.route('/attendance')
@login_required
def view_attendance():
    if current_user.role != 'student':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    attendance_records = Attendance.query.filter_by(student_id=current_user.id).order_by(Attendance.date.desc()).all()

    total = len(attendance_records)
    present = len([a for a in attendance_records if a.status == 'present'])
    absent = len([a for a in attendance_records if a.status == 'absent'])
    late = len([a for a in attendance_records if a.status == 'late'])

    return render_template('attendance.html',
                           attendance_records=attendance_records,
                           total=total,
                           present=present,
                           absent=absent,
                           late=late)


@app.route('/notes')
@login_required
def view_notes():
    notes = Note.query.order_by(Note.uploaded_at.desc()).all()
    return render_template('notes.html', notes=notes)


@app.route('/assignments')
@login_required
def view_assignments():
    assignments = Assignment.query.order_by(Assignment.uploaded_at.desc()).all()
    return render_template('assignments.html', assignments=assignments)


@app.route('/announcements')
def view_announcements():
    announcements = Announcement.query.filter(
        Announcement.expires_at.is_(None) | (Announcement.expires_at > datetime.utcnow())
    ).order_by(Announcement.created_at.desc()).all()

    speaker_status = tts_service.get_speaker_status()

    return render_template(
        'announcements.html',
        announcements=announcements,
        speaker_status=speaker_status
    )


@app.route('/events')
@login_required
def view_events():
    events = Event.query.filter_by(is_active=True).order_by(Event.event_date.asc()).all()
    return render_template('events.html', events=events)


# ==================== FACULTY ATTENDANCE ====================

@app.route('/faculty/attendance', methods=['GET', 'POST'])
@login_required
def faculty_attendance():
    if current_user.role != 'faculty':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    courses = get_all_courses()

    if request.method == 'POST':
        course = request.form.get('course')
        date_str = request.form.get('date')
        if not course or not date_str:
            flash('Please select course and date', 'warning')
            return redirect(url_for('faculty_attendance'))
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format', 'danger')
            return redirect(url_for('faculty_attendance'))
        students = StudentProfile.query.filter_by(course=course).all()
        date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        existing_records = Attendance.query.filter(
        Attendance.course == course,
        Attendance.date >= date_start,
        Attendance.date < date_end    ).all()
        attendance_status = {rec.student_id: rec.status for rec in existing_records}
        return render_template('faculty_attendance.html',
                           students=students,
                           course=course,
                           date=date,
                           courses=courses,
                           attendance_status=attendance_status)

    return render_template('faculty_attendance.html', courses=courses)


@app.route('/faculty/mark_attendance', methods=['POST'])
@login_required
def mark_attendance():
    if current_user.role != 'faculty':
        return jsonify({'error': 'Access denied'}), 403

    student_id = request.form.get('student_id')
    course = request.form.get('course')
    status = request.form.get('status')
    date_str = request.form.get('date')

    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid date'}), 400

    existing = Attendance.query.filter_by(
        student_id=student_id,
        course=course,
        date=date
    ).first()

    if existing:
        return jsonify({'error': 'Attendance already marked for this student'}), 400

    attendance = Attendance(
        student_id=student_id,
        course=course,
        date=date,
        status=status,
        marked_by=current_user.id,
        marked_at=datetime.utcnow()
    )

    db.session.add(attendance)
    db.session.commit()

    student_attendance = Attendance.query.filter_by(student_id=student_id).all()
    total = len(student_attendance)
    present = len([a for a in student_attendance if a.status == 'present'])
    percentage = (present / total * 100) if total > 0 else 0

    student_profile = StudentProfile.query.filter_by(user_id=student_id).first()
    if student_profile:
        student_profile.attendance_percentage = percentage
        db.session.commit()

    return jsonify({'success': True, 'message': 'Attendance marked successfully'})


# ==================== CAMERA-BASED FACE ATTENDANCE ====================

@app.route('/faculty/start_camera_attendance', methods=['POST'])
@login_required
def start_camera_attendance():
    if current_user.role != 'faculty':
        return jsonify({'error': 'Access denied'}), 403

    course = request.form.get('course')
    if not course:
        return jsonify({'error': 'Course is required'}), 400

    session['attendance_course'] = course
    session['attendance_active'] = True
    session['faculty_id'] = current_user.id

    return jsonify({
        'success': True,
        'message': 'Camera attendance session started',
        'course': course
    })


@app.route('/faculty/camera_feed')
@login_required
def camera_feed():
    """Stream camera feed with face recognition overlay."""
    if current_user.role != 'faculty':
        return "Access denied", 403

    # ⚠️ IMPORTANT: Capture session values BEFORE entering the generator
    # The generator runs outside the request context, so `session` won't work inside.
    course = session.get('attendance_course')
    faculty_id = session.get('faculty_id', current_user.id)

    if not course:
        return "No active session", 400

    def generate_frames():
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            app.logger.error("Cannot open camera")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)

        recognized_in_session = set()
        last_recognition_time = {}
        recognition_cooldown = 3

        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    break

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                predictions = identify_faces_from_frame(rgb_small_frame)
                current_time = time.time()

                for name, (top, right, bottom, left) in predictions:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

                    display_name = name if name != "unknown" else "Unknown"
                    cv2.putText(frame, display_name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    if name != "unknown":
                        # Use LOCAL variables, not session
                        if name not in last_recognition_time or \
                                (current_time - last_recognition_time[name]) > recognition_cooldown:
                            last_recognition_time[name] = current_time
                            try:
                                # mark_attendance_by_face uses app context — must push it
                                with app.app_context():
                                    marked = mark_attendance_by_face(name, course, faculty_id)
                                if marked:
                                    recognized_in_session.add(name)
                            except Exception as e:
                                app.logger.error(f"Error marking attendance: {str(e)}")

                cv2.putText(frame, f"Course: {course}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Recognized: {len(recognized_in_session)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        finally:
            camera.release()

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/faculty/stop_camera_attendance', methods=['POST'])
@login_required
def stop_camera_attendance():
    if current_user.role != 'faculty':
        return jsonify({'error': 'Access denied'}), 403

    course = session.get('attendance_course')
    session['attendance_active'] = False

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    marked_count = 0
    if course:
        marked_count = Attendance.query.filter(
            Attendance.course == course,
            Attendance.date >= today_start,
            Attendance.date < today_end,
            Attendance.marked_by == current_user.id
        ).count()

    session.pop('attendance_course', None)
    session.pop('faculty_id', None)

    return jsonify({
        'success': True,
        'message': f'Camera session stopped. {marked_count} students marked present.',
        'marked_count': marked_count
    })


@app.route('/faculty/attendance_status')
@login_required
def attendance_status():
    if current_user.role != 'faculty':
        return jsonify({'error': 'Access denied'}), 403

    course = session.get('attendance_course')
    is_active = session.get('attendance_active', False)

    if not course:
        return jsonify({
            'active': False,
            'course': None,
            'marked_count': 0,
            'marked_students': []
        })

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    marked = db.session.query(Attendance, User).join(
        User, Attendance.student_id == User.id
    ).filter(
        Attendance.course == course,
        Attendance.date >= today_start,
        Attendance.date < today_end
    ).all()

    marked_students = []
    for att, user in marked:
        profile = StudentProfile.query.filter_by(user_id=user.id).first()
        time_str = att.marked_at.strftime('%H:%M:%S') if att.marked_at else 'N/A'
        marked_students.append({
            'name': user.full_name,
            'student_id': profile.student_id if profile else 'N/A',
            'status': att.status,
            'time': time_str
        })

    return jsonify({
        'active': is_active,
        'course': course,
        'marked_count': len(marked_students),
        'marked_students': marked_students
    })


# ==================== FACULTY UPLOADS ====================

@app.route('/faculty/upload_note', methods=['POST'])
@login_required
def upload_note():
    if current_user.role != 'faculty':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('faculty_dashboard'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('faculty_dashboard'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'notes', filename)
        file.save(file_path)

        note = Note(
            title=request.form.get('title'),
            description=request.form.get('description'),
            file_path=file_path,
            course=request.form.get('course'),
            uploaded_by=current_user.id
        )

        db.session.add(note)
        db.session.commit()
        flash('Note uploaded successfully!', 'success')
    else:
        flash('Invalid file type', 'danger')

    return redirect(url_for('faculty_dashboard'))


@app.route('/faculty/upload_assignment', methods=['POST'])
@login_required
def upload_assignment():
    if current_user.role != 'faculty':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('faculty_dashboard'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('faculty_dashboard'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'assignments', filename)
        file.save(file_path)

        deadline_str = request.form.get('deadline')
        deadline = None
        if deadline_str:
            try:
                deadline = datetime.strptime(deadline_str, '%Y-%m-%dT%H:%M')
            except ValueError:
                deadline = datetime.strptime(deadline_str, '%Y-%m-%d %H:%M')

        assignment = Assignment(
            title=request.form.get('title'),
            description=request.form.get('description'),
            file_path=file_path,
            course=request.form.get('course'),
            deadline=deadline,
            uploaded_by=current_user.id
        )

        db.session.add(assignment)
        db.session.commit()
        flash('Assignment uploaded successfully!', 'success')
    else:
        flash('Invalid file type', 'danger')

    return redirect(url_for('faculty_dashboard'))


@app.route('/faculty/post_announcement', methods=['POST'])
@login_required
def post_announcement():
    if current_user.role not in ('faculty', 'admin'):
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    title = request.form.get('title')
    content = request.form.get('content')
    category = request.form.get('category')
    priority = request.form.get('priority', 'normal')
    expires_str = request.form.get('expires_at')
    expires_at = None
    if expires_str:
        try:
            expires_at = datetime.strptime(expires_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            expires_at = datetime.strptime(expires_str, '%Y-%m-%d %H:%M')

    announcement = Announcement(
        title=title,
        content=content,
        category=category,
        priority=priority,
        created_by=current_user.id,
        expires_at=expires_at
    )

    db.session.add(announcement)
    db.session.flush()
    db.session.commit()

    if content:
        try:
            broadcast_ok = tts_service.broadcast_announcement(
                title=title,
                content=content,
                priority=priority,
                announcement_id=announcement.id
            )
            if broadcast_ok:
                announcement.is_voice_announced = True
                announcement.voice_announce_time = datetime.utcnow()
                db.session.commit()
                flash('Announcement posted and broadcast through the department speaker!', 'success')
            else:
                flash('Announcement posted, but the department speaker could not be used.', 'warning')
        except Exception as e:
            app.logger.error(f'Announcement broadcast error: {str(e)}')
            flash('Announcement posted, but voice broadcast failed.', 'warning')
    else:
        flash('Announcement posted successfully!', 'success')

    seen = AnnouncementSeen(user_id=current_user.id, announcement_id=announcement.id)
    db.session.add(seen)
    db.session.commit()

    return redirect(url_for('faculty_dashboard'))


# ==================== ADMIN FEATURES ====================

@app.route('/admin/users')
@login_required
def manage_users():
    if current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    users = User.query.all()
    return render_template('manage_users.html', users=users)


@app.route('/admin/user/activate/<int:user_id>')
@login_required
def activate_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    user = User.query.get_or_404(user_id)
    user.is_active = True
    db.session.commit()
    flash(f'User {user.username} activated', 'success')
    return redirect(url_for('manage_users'))


@app.route('/admin/user/deactivate/<int:user_id>')
@login_required
def deactivate_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('Cannot deactivate yourself', 'danger')
        return redirect(url_for('manage_users'))

    user.is_active = False
    db.session.commit()
    flash(f'User {user.username} deactivated', 'warning')
    return redirect(url_for('manage_users'))


@app.route('/admin/create_event', methods=['POST'])
@login_required
def create_event():
    if current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    title = request.form.get('title')
    description = request.form.get('description')
    event_date_str = request.form.get('event_date')
    location = request.form.get('location')

    try:
        event_date = datetime.strptime(event_date_str, '%Y-%m-%dT%H:%M')
    except ValueError:
        event_date = datetime.strptime(event_date_str, '%Y-%m-%d %H:%M')

    event = Event(
        title=title,
        description=description,
        event_date=event_date,
        location=location,
        organized_by=current_user.id
    )

    db.session.add(event)
    db.session.commit()
    flash('Event created successfully!', 'success')
    return redirect(url_for('admin_dashboard'))


# ==================== VOICE ANNOUNCEMENT API ====================

@app.route('/api/announcements/voice/<int:announcement_id>')
def get_voice_announcement(announcement_id):
    announcement = Announcement.query.get_or_404(announcement_id)

    if not announcement.is_voice_announced:
        return jsonify({'error': 'No voice announcement available'}), 404

    try:
        audio_file = tts_service.text_to_speech(
            f"Announcement: {announcement.title}. {announcement.content}",
            f'announcement_{announcement_id}',
            use_gtts=True
        )
        audio_url = tts_service.get_audio_url(audio_file)
        return jsonify({'audio_url': audio_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/speaker/status')
def speaker_status():
    return jsonify(tts_service.get_speaker_status())


# ==================== NOTIFICATIONS API ====================

@app.route('/api/notifications')
@login_required
def get_notifications():
    unseen = Announcement.query.filter(
        ~AnnouncementSeen.query.filter(
            AnnouncementSeen.user_id == current_user.id,
            AnnouncementSeen.announcement_id == Announcement.id
        ).exists()
    ).filter(
        Announcement.expires_at.is_(None) | (Announcement.expires_at > datetime.utcnow())
    ).all()

    notifications = []
    for ann in unseen:
        notifications.append({
            'id': ann.id,
            'title': ann.title,
            'content': ann.content[:100],
            'created_at': ann.created_at.strftime('%Y-%m-%d %H:%M'),
            'priority': ann.priority
        })

    return jsonify(notifications)


@app.route('/api/announcements/seen/<int:announcement_id>', methods=['POST'])
@login_required
def mark_announcement_seen(announcement_id):
    announcement = Announcement.query.get_or_404(announcement_id)

    seen = AnnouncementSeen.query.filter_by(
        user_id=current_user.id,
        announcement_id=announcement_id
    ).first()

    if not seen:
        seen = AnnouncementSeen(
            user_id=current_user.id,
            announcement_id=announcement_id
        )
        db.session.add(seen)
        db.session.commit()

    return jsonify({'success': True})


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500


if __name__ == '__main__':
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'notes'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'assignments'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'face_data'), exist_ok=True)
    os.makedirs('database', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    init_db()

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)