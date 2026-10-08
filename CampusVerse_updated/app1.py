import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_socketio import SocketIO, emit, join_room
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_migrate import Migrate
from sqlalchemy import func
from config import Config
from fsm import PetState, FSM
from behavior import AdaptiveBehavior
from nlp import LocalNLP
import train_faces as knn_train
import cv2
import face_recognition
import numpy as np
import pandas as pd
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ------------------ DB MODELS ------------------

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)  # store hashed in production!
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    pet = db.relationship("Pet", uselist=False, back_populates="owner")

class Pet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    name = db.Column(db.String(64), default="Buddy")
    personality = db.Column(db.String(32), default="playful")

    # numeric state
    energy = db.Column(db.Float, default=70.0)
    hunger = db.Column(db.Float, default=30.0)
    happiness = db.Column(db.Float, default=60.0)
    mood = db.Column(db.String(32), default="neutral")
    last_update_ts = db.Column(db.Float, default=0.0)

    xp = db.Column(db.Integer, default=0)
    level = db.Column(db.Integer, default=1)

    owner = db.relationship("User", back_populates="pet")
    interactions = db.relationship("Interaction", back_populates="pet", cascade="all, delete-orphan")
    logs = db.relationship("ConversationLog", back_populates="pet", cascade="all, delete-orphan")

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pet_id = db.Column(db.Integer, db.ForeignKey("pet.id"))
    action = db.Column(db.String(32))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    delta_happiness = db.Column(db.Float, default=0.0)
    delta_energy = db.Column(db.Float, default=0.0)
    delta_hunger = db.Column(db.Float, default=0.0)
    reward = db.Column(db.Float, default=0.0)

    pet = db.relationship("Pet", back_populates="interactions")

class ConversationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pet_id = db.Column(db.Integer, db.ForeignKey("pet.id"))
    role = db.Column(db.String(16))  # "user" or "pet"
    text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    pet = db.relationship("Pet", back_populates="logs")

class Achievement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    title = db.Column(db.String(128))
    details = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ------------------ HELPERS ------------------

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

def get_fsm_for(pet: Pet) -> FSM:
    state = PetState(
        mood=pet.mood,
        energy=pet.energy,
        hunger=pet.hunger,
        happiness=pet.happiness,
        personality=pet.personality,
        last_update_ts=pet.last_update_ts or 0.0,
    )
    return FSM(state)

def persist_state(pet: Pet, fsm: FSM):
    s = fsm.state
    pet.mood = s.mood
    pet.energy = s.energy
    pet.hunger = s.hunger
    pet.happiness = s.happiness
    pet.last_update_ts = s.last_update_ts
    db.session.commit()

# Instantiate modules
ab = AdaptiveBehavior()
nlp = LocalNLP(enabled=app.config["USE_LOCAL_NLP"], model_name=app.config["NLP_MODEL_NAME"])

# ------------------ ROUTES ------------------

@app.route("/")
@login_required
def home():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
@login_required
def dashboard():
    pet = current_user.pet
    if not pet:
        # init default pet
        pet = Pet(owner=current_user, name="Buddy", personality="playful")
        db.session.add(pet)
        db.session.commit()
    fsm = get_fsm_for(pet)
    state = fsm.to_dict()
    persist_state(pet, fsm)
    achievements = Achievement.query.filter_by(user_id=current_user.id).order_by(Achievement.created_at.desc()).all()
    return render_template("dashboard.html", pet=pet, state=state, achievements=achievements)

@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html", pet=current_user.pet)
def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def identify_faces(video_capture):

    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="./models/trained_model.clf")
            # print(predictions)

        process_this_frame = not process_this_frame

        face_names = []

        for name, (top, right, bottom, left) in predictions:

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            #identify1(frame, name, buf, buf_length, known_conf)
            print("Name==",name)

            face_names.append(name)

        if name=="unknown":
            print("Unknown")
            break
        buf[i] = face_names
        i = (i + 1) % buf_length


        # print(buf)


        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    print("Face names===",face_names)
    return face_names
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        user = User.query.filter_by(username=username).first()
        # NOTE: Replace with proper password hashing (Werkzeug) in production
        if user and user.password == password:
            cam = cv2.VideoCapture(0)
            name=identify_faces(cam)
            name=name[0]
            name=name.replace(']','')
            name=name.replace("'","")
            print("name==",name)
            print("name from db==",user)
            if name==username:
                login_user(user)
                return redirect(url_for("dashboard"))
            else:
                flash("Face Not Matched", "danger")
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        if User.query.filter_by(username=username).first():
            flash("Username already taken", "warning")
            return render_template("register.html")
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        img_counter = 0
        adno=int(0)
        DIR=f"./Dataset/{username}_{adno}"
        try:
            os.mkdir(DIR)
            print("Directory " , username ,  " Created ")
        except FileExistsError:
            print("Directory " , username ,  " already exists")
            img_counter = len(os.listdir(DIR))
        while(True):
            ret, frame = cam.read()
            cv2.imshow("Video", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            if k%256 == 27:
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = f"./Dataset/{username}_{adno}/opencv_frame_{img_counter}.png"
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
        cam.release()
        cv2.destroyAllWindows()
        os.mkdir("./Data/"+str(username))
        knn_train.trainer()
        cam = cv2.VideoCapture(0)
        name=identify_faces(cam)
        name=name[0]
        name=name.replace(']','')
        name=name.replace("'","")
        print("name==",name)
        print("name from db==",username)
        if name==username:
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Face Not Matched", "danger")

        
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# -------- REST APIs (interactions, state, conversation) --------

@app.post("/api/interaction")
@login_required
def api_interaction():
    data = request.get_json(force=True)
    action = data.get("action", "pet")
    pet = current_user.pet
    fsm = get_fsm_for(pet)
    before = fsm.state

    fsm.apply_action(action)
    after = fsm.state
    persist_state(pet, fsm)

    # compute reward & gamify
    state_dict = fsm.to_dict()
    reward = ab.compute_reward(action, state_dict)
    pet.xp += max(1, int(5 + reward * 5))
    if pet.xp >= pet.level * 100:
        pet.level += 1
        db.session.add(Achievement(user_id=current_user.id, title=f"Level {pet.level}!", details="You leveled up!"))
    db.session.commit()

    interaction = Interaction(
        pet=pet, action=action,
        delta_happiness=after.happiness - before.happiness,
        delta_energy=after.energy - before.energy,
        delta_hunger=after.hunger - before.hunger,
        reward=reward
    )
    db.session.add(interaction); db.session.commit()

    # push realtime update
    socketio.emit("state_update", {"state": state_dict, "xp": pet.xp, "level": pet.level}, room=f"user-{current_user.id}")
    return jsonify(success=True, state=state_dict, xp=pet.xp, level=pet.level)

@app.get("/api/state")
@login_required
def api_state():
    pet = current_user.pet
    fsm = get_fsm_for(pet)
    state = fsm.to_dict()
    persist_state(pet, fsm)
    return jsonify(state=state, xp=pet.xp, level=pet.level)

@app.post("/api/converse")
@login_required
def api_converse():
    data = request.get_json(force=True)
    user_text = data.get("text", "")

    pet = current_user.pet
    fsm = get_fsm_for(pet)
    state = fsm.to_dict()

    # save user message
    db.session.add(ConversationLog(pet=pet, role="user", text=user_text))
    db.session.commit()

    # Try local model, otherwise rule-based response:
    prompt = f"You are a friendly virtual pet named {pet.name}. User says: {user_text}\nPet:"
    pet_reply = nlp.generate(prompt) or ab.select_response(user_text, state)

    db.session.add(ConversationLog(pet=pet, role="pet", text=pet_reply))
    db.session.commit()

    # Slight happiness bump for conversation
    fsm.apply_action("pet")
    persist_state(pet, fsm)

    payload = {"reply": pet_reply, "state": fsm.to_dict()}
    socketio.emit("chat_reply", payload, room=f"user-{current_user.id}")
    return jsonify(payload)

# ------------------ SOCKET.IO ------------------

@socketio.on("connect")
def on_connect():
    if current_user.is_authenticated:
        join_room(f"user-{current_user.id}")
        emit("connected", {"message": "Connected to realtime pet!"})

@socketio.on("interact")
def on_interact(data):
    # Proxy to REST so logic is centralized (optional)
    pass

# ------------------ MAIN ------------------

if __name__ == "__main__":
    # Initialize DB tables if not using migrations yet:
    with app.app_context():
        db.create_all()
    socketio.run(app, host="0.0.0.0", port=5000)
