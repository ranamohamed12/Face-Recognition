import PIL
import streamlit as st
import pickle
import numpy as np
import face_recognition
from PIL import Image
import tempfile

st.set_page_config(page_title="FaceSearcher.ai", layout="centered")
st.title("Face Detection and Recognition")
upload_img = st.file_uploader("Image", type=["jpg", "png", "jpeg"])
print("Load encode file")

# def load_image_file(file, mode='RGB'):
#     im = PIL.Image.open(file)
#     if mode:
#         im = im.convert(mode)
#     return np.array(im)

# Load the encoding file
try:
    with open("test_encodings.pkl", "rb") as f:
        encodings, labels = pickle.load(f)
    st.success("Encoding file loaded successfully")
except FileNotFoundError:
    st.error("Encoding file not found")
    st.stop()

# If an image is uploaded
if upload_img is not None:
    # Load image with face_recognition

    upload_img = Image.open(upload_img)
    if upload_img.mode not in ("RGB","L"):
        upload_img = upload_img.convert("RGB")
    img_np = np.array(upload_img)

    # Detect face locations and encodings

    face_locations = face_recognition.face_locations(img_np)  # Use NumPy array
    face_encodings = face_recognition.face_encodings(img_np, face_locations)  # Use NumPy array

    if len(face_encodings) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.image(upload_img, caption="Uploaded image", use_column_width=True)

        # Iterate over detected faces and compare them to the stored encodings
        for i, face_encoding in enumerate(face_encodings):
            distances = face_recognition.face_distance(encodings, face_encoding)
            match_distance = np.argmin(distances)

            if distances[match_distance] < 0.40:
                with col2:
                    st.write(f"Face {i + 1}: Match Found - {labels[match_distance]}")
                    st.write(f"Confidence: {round((1 - distances[match_distance]) * 100, 2)}%")
            else:
                with col2:
                    st.write(f"Face {i + 1}: No match found.")
                    st.write(f"Closest match distance: {round(distances[match_distance] * 100, 2)}%")
    else:
        st.warning("No face detected in the uploaded image.")
else:
    st.warning("Please upload an image.")
