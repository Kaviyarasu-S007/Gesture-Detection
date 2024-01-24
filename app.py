import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("D:\Internship\Prodigy\Gesture Detection\keras_model.h5", compile=False)

# Load the labels
class_names = open("D:\Internship\Prodigy\Gesture Detection\labels.txt", "r").readlines()

# Function to make predictions
def predict_gesture(image_path, progress_bar):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    
    try:
        # Attempt to open the image
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
    except Exception as e:
        st.error(f"Error opening or resizing image: {e}")
        return None, None

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Update progress bar with confidence score
    progress_bar.progress(int(confidence_score * 100))

    return class_name[2:], confidence_score

# Streamlit app
def main():
    st.title("Gesture Detection App")

    # Specify the image path
    image_path = "D:\Internship\Prodigy\Gesture Detection\class15.jpg"

    # Display the guide image
    guide_image = Image.open("guide.png").convert("RGB")
    st.image(guide_image, caption="Guide Image", use_column_width=True)

    # Display the image from the specified path
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="Image from Path", use_column_width=True)

    # Progress bar
    progress_bar = st.progress(0)

    prediction, confidence = predict_gesture(image_path, progress_bar)

    if prediction is not None and confidence is not None:
        # Display the prediction and confidence score
        st.success(f"Prediction: {prediction}, Confidence Score: {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
