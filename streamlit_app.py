import os
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils import shuffle

# Set the path for training data
train_data_dir = r"C:\Users\Sameer Gupta\Desktop\demo\train"

features = []
labels = []

# Load the EfficientNetB0 and ResNet50 models without the top layers
base_model_efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Iterate over the training data directory
for class_name in os.listdir(train_data_dir):
    class_dir = os.path.join(train_data_dir, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Extract features from the pre-trained models
            features_efficientnet = base_model_efficientnet.predict(img_array).flatten()
            features_resnet = base_model_resnet.predict(img_array).flatten()
            combined_features = np.concatenate([features_efficientnet, features_resnet])

            features.append(combined_features)
            labels.append(class_name)

# Convert the features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Shuffle the data
features, labels = shuffle(features, labels, random_state=42)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Create a Streamlit app
st.title("Image Classification with Random Forest and Transfer Learning")
st.write("Upload an image to classify its class using Random Forest and pre-trained models.")

# Function to prepare a single image for prediction
def prepare_image_for_prediction(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# File uploader for user input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare the uploaded image for prediction
    uploaded_image_array = np.array(image.resize((224, 224)))
    uploaded_image_array = np.expand_dims(uploaded_image_array, axis=0)
    uploaded_image_array = preprocess_input(uploaded_image_array)

    # Extract features from the pre-trained models for the uploaded image
    uploaded_image_features_efficientnet = base_model_efficientnet.predict(uploaded_image_array).flatten()
    uploaded_image_features_resnet = base_model_resnet.predict(uploaded_image_array).flatten()
    combined_uploaded_image_features = np.concatenate([uploaded_image_features_efficientnet, uploaded_image_features_resnet])

    # Make a prediction using the trained Random Forest classifier
    predicted_class_rf = rf_classifier.predict([combined_uploaded_image_features])[0]
    st.write(f"Random Forest Predicted class: {predicted_class_rf}")
