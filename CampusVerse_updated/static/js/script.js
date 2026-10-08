// Global variables
let notificationInterval = null;

// Document ready
$(document).ready(function () {
    initNotifications();

    if (notificationInterval) {
        clearInterval(notificationInterval);
    }
    notificationInterval = setInterval(initNotifications, 30000);

    $('#notificationBell').click(function (e) {
        e.preventDefault();
        showNotifications();
    });
});

// Initialize notifications
function initNotifications() {
    if (typeof current_user !== 'undefined' && current_user.is_authenticated) {
        $.ajax({
            url: '/api/notifications',
            type: 'GET',
            success: function (data) {
                updateNotificationBadge(data.length);
                updateNotificationList(data);
            },
            error: function () {
                console.error('Error fetching notifications');
            }
        });
    }
}

// Update notification badge
function updateNotificationBadge(count) {
    const badge = $('#notificationBadge');
    if (count > 0) {
        badge.text(count);
        badge.show();
    } else {
        badge.hide();
    }
}

// Update notification list
function updateNotificationList(notifications) {
    const list = $('#notificationList');
    if (notifications.length === 0) {
        list.html('<div class="text-center text-muted"><i class="fas fa-check-circle"></i> No new notifications</div>');
        return;
    }

    let html = '';
    notifications.forEach(function (notification) {
        const priorityClass = notification.priority === 'high' ? 'border-danger' : 'border-info';
        html += `
            <div class="alert alert-${notification.priority === 'high' ? 'danger' : 'info'} alert-dismissible fade show border-start border-4 ${priorityClass}">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h6>${notification.title}</h6>
                        <p class="mb-0">${notification.content}</p>
                        <small class="text-muted">${notification.created_at}</small>
                    </div>
                    <button class="btn-close" onclick="markNotificationSeen(${notification.id})"></button>
                </div>
                ${notification.priority === 'high' ? '<div class="mt-1"><span class="badge bg-danger">Important</span></div>' : ''}
            </div>
        `;
    });

    list.html(html);
}

// Mark notification as seen
function markNotificationSeen(announcementId) {
    $.ajax({
        url: '/api/announcements/seen/' + announcementId,
        type: 'POST',
        success: function () {
            initNotifications();
        },
        error: function () {
            console.error('Error marking notification as seen');
        }
    });
}

// Show notifications modal
function showNotifications() {
    const modal = new bootstrap.Modal(document.getElementById('notificationModal'));
    modal.show();
}

// Show toast notification
function showToast(title, message, type = 'info') {
    const toastHtml = `
        <div class="position-fixed bottom-0 end-0 p-3" style="z-index: 11">
            <div class="toast show" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header bg-${type} text-white">
                    <strong class="me-auto">${title}</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        </div>
    `;

    $('body').append(toastHtml);
    setTimeout(function () {
        $('.toast').remove();
    }, 5000);
}

// Validate form inputs
function validateForm(form) {
    let isValid = true;
    const inputs = form.find('input[required], select[required], textarea[required]');

    inputs.each(function () {
        const input = $(this);
        if (!input.val() || input.val().trim() === '') {
            input.addClass('is-invalid');
            isValid = false;
        } else {
            input.removeClass('is-invalid');
        }
    });

    return isValid;
}

// File validation
function validateFile(input, allowedTypes = ['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png']) {
    const file = input.files[0];
    if (!file) return false;

    const extension = file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(extension)) {
        showToast('Error', 'Invalid file type. Allowed: ' + allowedTypes.join(', '), 'danger');
        return false;
    }

    if (file.size > 10 * 1024 * 1024) {
        showToast('Error', 'File size must be less than 10MB', 'danger');
        return false;
    }

    return true;
}

// Date formatting helper
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Truncate text
function truncateText(text, length = 100) {
    if (!text) return '';
    if (text.length <= length) return text;
    return text.substring(0, length) + '...';
}

// Get status badge HTML
function getStatusBadge(status) {
    const badges = {
        'present': '<span class="badge bg-success">Present</span>',
        'absent': '<span class="badge bg-danger">Absent</span>',
        'late': '<span class="badge bg-warning">Late</span>',
        'active': '<span class="badge bg-success">Active</span>',
        'inactive': '<span class="badge bg-danger">Inactive</span>'
    };
    return badges[status] || '<span class="badge bg-secondary">Unknown</span>';
}

// Get priority badge HTML
function getPriorityBadge(priority) {
    const badges = {
        'high': '<span class="badge bg-danger">High</span>',
        'normal': '<span class="badge bg-primary">Normal</span>',
        'low': '<span class="badge bg-secondary">Low</span>'
    };
    return badges[priority] || badges['normal'];
}

// Play voice announcement
function playVoiceAnnouncement(announcementId) {
    $.ajax({
        url: '/api/announcements/voice/' + announcementId,
        type: 'GET',
        success: function (data) {
            if (data.audio_url) {
                const audio = new Audio(data.audio_url);
                audio.play();
                showToast('Voice', 'Playing announcement...', 'info');
            } else {
                showToast('Error', 'No voice announcement available', 'danger');
            }
        },
        error: function () {
            showToast('Error', 'Failed to play voice announcement', 'danger');
        }
    });
}

// Confirm action
function confirmAction(message) {
    return confirm(message || 'Are you sure?');
}

// Export functions
window.formatDate = formatDate;
window.truncateText = truncateText;
window.getStatusBadge = getStatusBadge;
window.getPriorityBadge = getPriorityBadge;
window.playVoiceAnnouncement = playVoiceAnnouncement;
window.confirmAction = confirmAction;
window.showToast = showToast;
window.validateFile = validateFile;