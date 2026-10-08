import math
import os
import pickle

import cv2
import numpy as np
import face_recognition
from sklearn import neighbors
from tqdm import tqdm


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def get_image_files(folder):
    """Get supported image files from a person's folder."""

    if not os.path.isdir(folder):
        return []

    image_files = []

    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            continue

        extension = filename.rsplit(".", 1)[-1].lower()

        if extension in ALLOWED_EXTENSIONS:
            image_files.append(filepath)

    return image_files


def load_image_for_face_recognition(image_path):
    """
    Load image using OpenCV and convert it to the exact
    format expected by dlib:

        uint8
        3 channels
        RGB
        contiguous NumPy array
    """

    # IMREAD_COLOR guarantees 8-bit BGR
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("OpenCV could not read the image")

    # Check datatype
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # OpenCV uses BGR
    # face_recognition/dlib expects RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Force contiguous memory
    image = np.ascontiguousarray(image, dtype=np.uint8)

    # Final validation
    if image.dtype != np.uint8:
        raise ValueError(
            f"Invalid dtype: {image.dtype}"
        )

    if len(image.shape) != 3:
        raise ValueError(
            f"Invalid image dimensions: {image.shape}"
        )

    if image.shape[2] != 3:
        raise ValueError(
            f"Invalid number of channels: {image.shape[2]}"
        )

    return image


def train(
    train_dir,
    model_save_path=None,
    n_neighbors=None,
    knn_algo="ball_tree",
    verbose=False
):
    """
    Train KNN face recognition model.

    Dataset structure:

    Dataset/
    ├── sanjeev_0/
    │   ├── opencv_frame_0.png
    │   ├── opencv_frame_1.png
    │   └── ...
    ├── person2_0/
    │   └── ...
    """

    X = []
    y = []

    # ---------------------------------------------------------
    # Check dataset
    # ---------------------------------------------------------

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Dataset folder does not exist: {train_dir}"
        )

    if not os.path.isdir(train_dir):
        raise ValueError(
            f"Dataset path is not a directory: {train_dir}"
        )

    # ---------------------------------------------------------
    # Find people
    # ---------------------------------------------------------

    people = []

    for item in sorted(os.listdir(train_dir)):

        person_path = os.path.join(train_dir, item)

        if os.path.isdir(person_path):
            people.append(item)

    if not people:
        raise ValueError(
            "No person folders found inside Dataset."
        )

    print()
    print("=" * 70)
    print("FACE RECOGNITION TRAINING")
    print("=" * 70)
    print(
        "Dataset directory :",
        os.path.abspath(train_dir)
    )
    print(
        "People found      :",
        len(people)
    )
    print()

    # ---------------------------------------------------------
    # Process people
    # ---------------------------------------------------------

    for class_dir in tqdm(people, desc="People"):

        person_folder = os.path.join(
            train_dir,
            class_dir
        )

        image_files = get_image_files(person_folder)

        if not image_files:

            print(
                f"\n[WARNING] No images found for "
                f"{class_dir}"
            )

            continue

        print(
            f"\nProcessing '{class_dir}' "
            f"({len(image_files)} images)"
        )

        # -----------------------------------------------------
        # Process images
        # -----------------------------------------------------

        for img_path in image_files:

            try:

                # ---------------------------------------------
                # Load using OpenCV
                # ---------------------------------------------

                image = load_image_for_face_recognition(
                    img_path
                )

                if verbose:

                    print(
                        f"\nImage: {img_path}"
                    )

                    print(
                        "Shape:",
                        image.shape
                    )

                    print(
                        "Dtype:",
                        image.dtype
                    )

                    print(
                        "Contiguous:",
                        image.flags["C_CONTIGUOUS"]
                    )

                # ---------------------------------------------
                # Detect faces
                # ---------------------------------------------

                boxes = face_recognition.face_locations(
                    image,
                    number_of_times_to_upsample=1,
                    model="hog"
                )

                # ---------------------------------------------
                # Check number of faces
                # ---------------------------------------------

                if len(boxes) == 0:

                    print(
                        f"[SKIPPED] No face found: "
                        f"{img_path}"
                    )

                    continue

                if len(boxes) > 1:

                    print(
                        f"[SKIPPED] Multiple faces found: "
                        f"{img_path}"
                    )

                    continue

                # ---------------------------------------------
                # Generate encoding
                # ---------------------------------------------

                encodings = face_recognition.face_encodings(
                    image,
                    known_face_locations=boxes
                )

                if not encodings:

                    print(
                        f"[SKIPPED] Face encoding failed: "
                        f"{img_path}"
                    )

                    continue

                encoding = encodings[0]

                # ---------------------------------------------
                # Save encoding
                # ---------------------------------------------

                X.append(encoding)

                # Example:
                #
                # sanjeev_0
                #
                # becomes:
                #
                # sanjeev
                #

                person_name = class_dir.split("_")[0]

                y.append(person_name)

                print(
                    f"[OK] Face registered from: "
                    f"{os.path.basename(img_path)} "
                    f"-> {person_name}"
                )

            except Exception as e:

                print()
                print(
                    "[ERROR] Could not process:"
                )

                print(
                    "        ",
                    img_path
                )

                print(
                    "Reason:",
                    str(e)
                )

                print(
                    "Skipping this image..."
                )

                continue

    # ---------------------------------------------------------
    # Training summary
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("TRAINING DATA SUMMARY")
    print("-" * 70)

    print(
        "Valid face encodings :",
        len(X)
    )

    print(
        "People/classes       :",
        len(set(y))
    )

    print()

    # ---------------------------------------------------------
    # No faces
    # ---------------------------------------------------------

    if len(X) == 0:

        raise ValueError(
            "\nNo valid face encodings were found.\n\n"
            "Possible reasons:\n"
            "1. Images do not contain a detectable face.\n"
            "2. Images are too dark/blurry.\n"
            "3. The camera images were captured incorrectly.\n"
            "4. dlib/face_recognition installation is incompatible.\n"
        )

    # ---------------------------------------------------------
    # Determine neighbors
    # ---------------------------------------------------------

    if n_neighbors is None:

        n_neighbors = int(
            round(math.sqrt(len(X)))
        )

        n_neighbors = max(
            1,
            n_neighbors
        )

        n_neighbors = min(
            n_neighbors,
            len(X)
        )

    else:

        n_neighbors = max(
            1,
            int(n_neighbors)
        )

        n_neighbors = min(
            n_neighbors,
            len(X)
        )

    print(
        "KNN neighbors:",
        n_neighbors
    )

    # ---------------------------------------------------------
    # Create classifier
    # ---------------------------------------------------------

    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm=knn_algo,
        weights="distance"
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print(
        "Training KNN classifier..."
    )

    knn_clf.fit(
        X,
        y
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    if model_save_path:

        model_directory = os.path.dirname(
            os.path.abspath(model_save_path)
        )

        os.makedirs(
            model_directory,
            exist_ok=True
        )

        with open(
            model_save_path,
            "wb"
        ) as f:

            pickle.dump(
                knn_clf,
                f
            )

        print()
        print(
            "Model saved successfully:"
        )

        print(
            os.path.abspath(
                model_save_path
            )
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()

    return knn_clf


def trainer():

    print()
    print(
        "Starting face recognition training..."
    )
    print()

    classifier = train(
        train_dir="./Dataset",
        model_save_path="./models/trained_model.clf",
        n_neighbors=3,
        knn_algo="ball_tree",
        verbose=True
    )

    print(
        "Face recognition model is ready."
    )

    return classifier


# if __name__ == "__main__":
#     trainer()