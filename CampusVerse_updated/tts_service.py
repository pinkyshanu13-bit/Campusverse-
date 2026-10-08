import os
import sys
import threading
import hashlib
from datetime import datetime

import pyttsx3
from gtts import gTTS
import playsound


class TTSService:
    """Text-to-speech and automatic department-speaker broadcasting."""

    def __init__(self):
        self.engine = None
        self.voice_enabled = True
        self.audio_cache = {}
        self.audio_dir = 'static/audio'
        self.playback_lock = threading.Lock()

        os.makedirs(self.audio_dir, exist_ok=True)

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)

            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    name = getattr(voice, 'name', '') or ''
                    if 'female' in name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
        except Exception as e:
            print(f"Warning: pyttsx3 initialization failed: {e}")
            self.engine = None

    def text_to_speech(self, text, identifier=None, use_gtts=True):
        """Generate an MP3 file and return its local path."""
        if not text:
            return None

        digest = hashlib.md5(text.encode('utf-8')).hexdigest()[:10]
        filename = (
            f"{identifier}_{digest}.mp3" if identifier
            else f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest}.mp3"
        )
        filepath = os.path.join(self.audio_dir, filename)

        if os.path.exists(filepath):
            self.audio_cache[filename] = filepath
            return filepath

        try:
            if use_gtts:
                try:
                    tts = gTTS(text=text, lang='en', slow=False)
                    tts.save(filepath)
                except Exception as online_error:
                    # Keep department announcements working if Internet TTS
                    # is unavailable and an offline pyttsx3 engine exists.
                    print(f"gTTS failed, using offline TTS: {online_error}")
                    if self.engine:
                        self.engine.save_to_file(text, filepath)
                        self.engine.runAndWait()
                    else:
                        raise
            elif self.engine:
                self.engine.save_to_file(text, filepath)
                self.engine.runAndWait()
            else:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(filepath)

            if not os.path.exists(filepath):
                raise RuntimeError("TTS did not create the audio file")

            self.audio_cache[filename] = filepath
            return filepath

        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return None

    def speak(self, text, async_mode=True):
        """Speak text through the computer's active/default audio output."""
        if not self.voice_enabled or not text:
            return False

        def speak_thread():
            try:
                with self.playback_lock:
                    if self.engine:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    else:
                        audio_file = self.text_to_speech(text, use_gtts=True)
                        if audio_file:
                            playsound.playsound(audio_file, block=True)
            except Exception as e:
                print(f"Error in speech: {e}")

        if async_mode:
            threading.Thread(target=speak_thread, daemon=True).start()
        else:
            speak_thread()
        return True

    def announce_announcement(self, announcement):
        """Generate audio for an Announcement model instance."""
        if not announcement or not announcement.content:
            return None

        text = f"Announcement: {announcement.title}. {announcement.content}"
        if announcement.priority == 'high':
            text = f"Important! {text}"

        return self.text_to_speech(text, f"announcement_{announcement.id}", use_gtts=True)

    def broadcast_announcement(self, title, content, priority='normal', announcement_id=None):
        """Generate TTS and automatically play it on the active system output."""
        if not content:
            return False

        text = f"Announcement: {title}. {content}"
        if priority == 'high':
            text = f"Important! {text}"

        identifier = f"announcement_{announcement_id}" if announcement_id else None
        audio_file = self.text_to_speech(text, identifier, use_gtts=True)

        if not audio_file:
            return False

        def play():
            try:
                with self.playback_lock:
                    playsound.playsound(audio_file, block=True)
            except Exception as e:
                print(f"Department speaker playback error: {e}")

        threading.Thread(target=play, daemon=True).start()
        return True

    def get_audio_url(self, audio_path):
        """Get a browser-safe URL for an audio file."""
        if audio_path and os.path.exists(audio_path):
            return f"/static/audio/{os.path.basename(audio_path)}"
        return None

    def get_speaker_status(self):
        """Report the active audio endpoint and whether it appears Bluetooth."""
        status = {
            'connected': False,
            'device_name': 'No active audio output detected',
            'is_bluetooth': False,
            'message': 'Department speaker not detected'
        }

        if sys.platform.startswith('win'):
            try:
                from pycaw.pycaw import AudioUtilities

                device = AudioUtilities.GetSpeakers()
                if device:
                    name = str(getattr(device, 'FriendlyName', '') or device).strip()
                    lower = name.lower()
                    bluetooth_terms = (
                        'bluetooth', 'wireless', 'headset', 'hands-free',
                        'airpods', 'buds', 'jbl', 'boat', 'sony', 'bose'
                    )
                    is_bluetooth = any(term in lower for term in bluetooth_terms)

                    status.update({
                        'connected': True,
                        'device_name': name,
                        'is_bluetooth': is_bluetooth,
                        'message': (
                            'Bluetooth department speaker connected'
                            if is_bluetooth
                            else 'Audio output connected'
                        )
                    })
                    return status

            except Exception as e:
                status['message'] = f'Audio output check unavailable: {e}'

        elif sys.platform.startswith(('linux', 'darwin')):
            status.update({
                'connected': True,
                'device_name': 'System default audio output',
                'message': 'System default audio output available'
            })

        return status

    def cleanup_old_audio(self, days=7):
        try:
            import time
            current_time = time.time()

            for filename in os.listdir(self.audio_dir):
                filepath = os.path.join(self.audio_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > days * 24 * 60 * 60:
                        os.remove(filepath)
                        self.audio_cache.pop(filename, None)
        except Exception as e:
            print(f"Error cleaning up audio: {e}")
