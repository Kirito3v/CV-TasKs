import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tf_keras.models import Sequential, load_model, Model
from tf_keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.tree import DecisionTreeClassifier

# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 10  # Number of epochs for CNN training
train_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/train'
validation_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test'

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,  # Increased rotation range
    width_shift_range=0.3,  # More significant width shift
    height_shift_range=0.3,  # More significant height shift
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'  # Fill missing pixels after transformations
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

validation_data = validation_datagen.flow_from_directory(
    validation_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define file paths for the models
cnn_model_path = 'cnn_model.h5'
dt_classifier_model_path = 'dt_classifier_model.pkl'

# Check if the CNN model exists and load it, else train it
if os.path.exists(cnn_model_path):
    print("Loading pre-trained CNN model...")
    cnn_model = load_model(cnn_model_path)
else:
    print("Training CNN model...")
    # Define and train the CNN model
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),  # This layer will ensure the output is a 1D feature vector
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Update with correct number of classes
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_data, epochs=epochs, validation_data=validation_data)
    
    # Save the trained CNN model
    cnn_model.save(cnn_model_path)
    print("CNN model saved.")

# Create an intermediate model to extract features before the output layer
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

def extract_or_load_features(model, data, feature_file, label_file):
    if os.path.exists(feature_file) and os.path.exists(label_file):
        # Load features and labels if files exist
        print(f"Loading features from {feature_file} and labels from {label_file}")
        features = np.load(feature_file)
        labels = np.load(label_file)
    else:
        # Extract and save features and labels if files do not exist
        print(f"Extracting features and saving to {feature_file} and {label_file}")
        features = []
        labels = []
        for _ in range(len(data)):
            imgs, lbls = next(data)
            feature_vectors = model.predict(imgs)  # Get feature vectors from CNN layers
            features.extend(feature_vectors)
            labels.extend(lbls)
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Save features and labels to files
        np.save(feature_file, features)
        np.save(label_file, labels)
    
    return features, labels

# Use the function for training data
train_features, train_labels = extract_or_load_features(
    feature_extractor, train_data, "train_features.npy", "train_labels.npy"
)

# Use the function for validation data
validation_features, validation_labels = extract_or_load_features(
    feature_extractor, validation_data, "validation_features.npy", "validation_labels.npy"
)

# Convert one-hot labels to class labels for validation data
validation_labels = np.argmax(validation_labels, axis=1)

# Check if the Decision Tree model exists and load it, else train it
if os.path.exists(dt_classifier_model_path):
    print("Loading pre-trained Decision Tree classifier model...")
    dt_classifier = joblib.load(dt_classifier_model_path)
else:
    print("Training Decision Tree classifier model...")
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(train_features, np.argmax(train_labels, axis=1))  # Ensure labels are in the right format
    
    # Save the trained Decision Tree model
    joblib.dump(dt_classifier, dt_classifier_model_path)
    print("Decision Tree classifier model saved.")

# Evaluate the Decision Tree classifier model on validation data
y_pred = dt_classifier.predict(validation_features)
accuracy = accuracy_score(validation_labels, y_pred)
print(f"Decision Tree classifier model accuracy : {accuracy * 100:.2f}%")

def classify_image(img_path):
    # Load and preprocess the new image
    img = load_img(img_path, target_size=(128, 128))  # Resize to match the CNN input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to match the training preprocessing

    # Extract features using the CNN model
    features = feature_extractor.predict(img_array)
    
    # Classify the features with the Decision Tree model
    prediction = dt_classifier.predict(features)
    
    # Interpret the prediction
    class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
    result = class_labels[prediction[0]]
    return result

# Test the function with a new image
img_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test/NonDemented/26 (66).jpg'
result = classify_image(img_path)
print(f"The image is classified as: {result}")
