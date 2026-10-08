from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import hashlib

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    # One-to-one relationships with student and faculty profiles
    student_profile = db.relationship(
        'StudentProfile',
        foreign_keys='StudentProfile.user_id',
        backref='user',
        uselist=False
    )

    faculty_profile = db.relationship(
        'FacultyProfile',
        foreign_keys='FacultyProfile.user_id',
        backref='user',
        uselist=False
    )

    # Announcements
    announcements_seen = db.relationship(
        'AnnouncementSeen',
        backref='user',
        lazy=True
    )

    announcements_created = db.relationship(
        'Announcement',
        foreign_keys='Announcement.created_by',
        backref='creator',
        lazy=True
    )

    def set_password(self, password):
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()

    def check_password(self, password):
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()


class StudentProfile(db.Model):
    __tablename__ = 'student_profiles'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    course = db.Column(db.String(50))
    year = db.Column(db.Integer)
    section = db.Column(db.String(10))
    face_encoding = db.Column(db.Text)
    attendance_percentage = db.Column(db.Float, default=0.0)


class FacultyProfile(db.Model):
    __tablename__ = 'faculty_profiles'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True)
    faculty_id = db.Column(db.String(20), unique=True, nullable=False)
    department = db.Column(db.String(50))
    designation = db.Column(db.String(50))


class Announcement(db.Model):
    __tablename__ = 'announcements'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))
    priority = db.Column(db.String(20), default='normal')
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    is_voice_announced = db.Column(db.Boolean, default=False)
    voice_announce_time = db.Column(db.DateTime)

    seen_by = db.relationship(
        'AnnouncementSeen',
        backref='announcement',
        lazy=True
    )


class AnnouncementSeen(db.Model):
    __tablename__ = 'announcement_seen'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    announcement_id = db.Column(db.Integer, db.ForeignKey('announcements.id'))
    seen_at = db.Column(db.DateTime, default=datetime.utcnow)


class Event(db.Model):
    __tablename__ = 'events'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    event_date = db.Column(db.DateTime, nullable=False)
    location = db.Column(db.String(200))
    organized_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    organizer = db.relationship(
        'User',
        foreign_keys=[organized_by],
        backref='events_organized'
    )


class Note(db.Model):
    __tablename__ = 'notes'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(300))
    course = db.Column(db.String(50))
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    uploader = db.relationship(
        'User',
        foreign_keys=[uploaded_by],
        backref='notes_uploaded'
    )


class Assignment(db.Model):
    __tablename__ = 'assignments'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(300))
    course = db.Column(db.String(50))
    deadline = db.Column(db.DateTime)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    uploader = db.relationship(
        'User',
        foreign_keys=[uploaded_by],
        backref='assignments_uploaded'
    )


class Attendance(db.Model):
    __tablename__ = 'attendance'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    course = db.Column(db.String(50))
    date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20))
    marked_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    marked_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships — use distinct backrefs to avoid conflicts with User
    student = db.relationship(
        'User',
        foreign_keys=[student_id],
        backref='attendance_as_student'
    )

    marker = db.relationship(
        'User',
        foreign_keys=[marked_by],
        backref='attendance_marked'
    )