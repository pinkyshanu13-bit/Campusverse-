import os
from datetime import datetime
import re

def allowed_file(filename, allowed_extensions=None):
    """Check if file extension is allowed"""
    if allowed_extensions is None:
        allowed_extensions = {'pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mp3'}
    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def format_datetime(dt, format_str='%Y-%m-%d %H:%M'):
    """Format datetime object to string"""
    if dt is None:
        return 'N/A'
    return dt.strftime(format_str)

def get_notifications_count(user_id):
    """Get count of unseen notifications for a user"""
    from models import Announcement, AnnouncementSeen, db
    from datetime import datetime
    
    unseen_count = Announcement.query.filter(
        ~AnnouncementSeen.query.filter(
            AnnouncementSeen.user_id == user_id,
            AnnouncementSeen.announcement_id == Announcement.id
        ).exists()
    ).filter(
        Announcement.expires_at.is_(None) | (Announcement.expires_at > datetime.utcnow())
    ).count()
    
    return unseen_count

def get_user_role(user):
    """Get user role with proper display name"""
    if user is None:
        return 'Unknown'
    
    role_map = {
        'admin': 'Administrator',
        'faculty': 'Faculty Member',
        'student': 'Student'
    }
    
    return role_map.get(user.role, user.role.capitalize())

def sanitize_input(text):
    """Sanitize user input to prevent XSS"""
    if text is None:
        return ''
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Escape special characters
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('"', '&quot;').replace("'", '&#x27;')
    
    return text

def generate_student_id(course, year):
    """Generate a unique student ID"""
    import random
    from datetime import datetime
    
    prefix = f"{course[:3].upper()}{year}"
    random_num = str(random.randint(1000, 9999))
    return f"{prefix}{random_num}"

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def get_file_size(file_path):
    """Get file size in human readable format"""
    try:
        size = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return 'Unknown'

def get_extension(filename):
    """Get file extension from filename"""
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''

def is_valid_date(date_string, format_str='%Y-%m-%d'):
    """Check if date string is valid"""
    try:
        datetime.strptime(date_string, format_str)
        return True
    except ValueError:
        return False

def truncate_text(text, length=100):
    """Truncate text to specified length"""
    if len(text) <= length:
        return text
    return text[:length] + '...'

def get_priority_badge(priority):
    """Get HTML badge for priority level"""
    badges = {
        'high': '<span class="badge badge-danger">High</span>',
        'normal': '<span class="badge badge-primary">Normal</span>',
        'low': '<span class="badge badge-secondary">Low</span>'
    }
    return badges.get(priority, badges['normal'])