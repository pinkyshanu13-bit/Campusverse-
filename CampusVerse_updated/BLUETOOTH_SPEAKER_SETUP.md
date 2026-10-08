# CampusVerse Department Speaker Setup

## How automatic broadcasting works

1. Run CampusVerse on the department computer.
2. Pair the Bluetooth speaker with that computer.
3. In Windows, select the Bluetooth speaker as the **default output device**.
4. Start the Flask application.
5. Open **Announcements**. The page shows the active audio output and Bluetooth detection.
6. Admin or Faculty posts an announcement.
7. CampusVerse generates the speech and automatically plays it through the computer's active audio output.
8. Students do not need to log in, open the announcement, or press Play.

## Important

The server-side application must run on the computer physically connected/paired to the department speaker. A Flask server running on a remote hosting service cannot directly play audio on a Bluetooth speaker in the college unless that machine has access to the speaker.

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, make sure the Bluetooth speaker is connected and selected as the Windows sound output.

The speaker status uses `pycaw` when available. It reports the active Windows render endpoint and marks it as Bluetooth when its device name indicates a Bluetooth/wireless output.
