import os
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tf_keras.models import Sequential, load_model
from tf_keras.preprocessing.image import ImageDataGenerator

# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 20  
train_path = 'dataset/train'
validation_path = 'dataset/validation'

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
    class_mode='binary',
    shuffle=False
)

validation_data = validation_datagen.flow_from_directory(
    validation_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Define file paths for the models
cnn_model_path = 'cnn_model.h5'
svm_model_path = 'svm_model.pkl'

# Check if the CNN model exists and load it, else train it
if os.path.exists(cnn_model_path):
    print("Loading pre-trained CNN model...")
    cnn_model = load_model(cnn_model_path)
else:
    print("Training CNN model...")
    # Define and train the CNN model (as in your original code)
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu')
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_data, epochs=epochs, validation_data=validation_data)
    
    # Save the trained CNN model
    cnn_model.save(cnn_model_path)
    print("CNN model saved.")


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
    cnn_model, train_data, "train_features.npy", "train_labels.npy"
)

# Use the function for validation data
validation_features, validation_labels = extract_or_load_features(
    cnn_model, validation_data, "validation_features.npy", "validation_labels.npy"
)

# Check if the SVM model exists and load it, else train it
if os.path.exists(svm_model_path):
    print("Loading pre-trained SVM model...")
    svm = joblib.load(svm_model_path)
else:
    print("Training SVM model...")
    svm = SVC(kernel='linear')
    svm.fit(train_features, train_labels)
    
    # Save the trained SVM model
    joblib.dump(svm, svm_model_path)
    print("SVM model saved.")

# Evaluate the best SVM model on validation data
y_pred = svm.predict(validation_features)
accuracy = accuracy_score(validation_labels, y_pred)
print(f"SVM model accuracy : {accuracy * 100:.2f}%")


# Select 10 random indices from the validation data
random_indices = random.sample(range(len(validation_labels)), 10)

# Get the corresponding images and labels
sample_images = []
real_labels = []
predicted_labels = []

# Ensure we can access specific samples
validation_data.reset()  # Reset the generator to start from the beginning

# Iterate through the validation generator to fetch specific samples
current_index = 0  # To keep track of overall index
for imgs, lbls in validation_data:
    batch_size = imgs.shape[0]  # Get the actual batch size (might vary for the last batch)

    for i in range(batch_size):
        if current_index in random_indices:
            sample_images.append(imgs[i])
            real_labels.append(lbls[i])
            predicted_labels.append(y_pred[current_index])
        
        current_index += 1

    if len(sample_images) == len(random_indices):
        break  # Stop when we’ve collected all required samples

# Plot the images with real and predicted labels
plt.figure(figsize=(12, 8))
for i, (image, real, predicted) in enumerate(zip(sample_images, real_labels, predicted_labels)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)  # Display the image
    plt.title(f"Real: {'Cat' if real == 0 else 'Dog'}\nPred: {'Cat' if predicted == 0 else 'Dog'}")
    plt.axis('off')

plt.tight_layout()
plt.show()