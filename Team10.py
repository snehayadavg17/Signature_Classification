import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Set up paths to signature data
data_dir = r"C:\Users\Guna Shekar\Desktop\Dataset"
signature_classes = ['sneha', 'sanjana', 'Invalid']  # Sub-folders in data_dir

# Load signature images and labels
X = []
y = []
for label, signature_class in enumerate(signature_classes):
    class_dir = os.path.join(data_dir, signature_class)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(img_gray)
            y.append(label)

# Preprocess signature images
def preprocess_signature_data(X):
    # Normalize pixel values
    X_norm = [img.astype(np.float32) / 255.0 for img in X]
    # Resize images to a standard size
    size = (200, 100)
    X_resized = [cv2.resize(img, size) for img in X_norm]
    # Flatten images into 1D arrays
    X_flat = [img.flatten() for img in X_resized]
    return np.array(X_flat)

X = preprocess_signature_data(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train an SVM model on the signature data
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = clf.score(X_test, y_test)
print('Test accuracy:', accuracy)

# Use the model to verify new signature images
img_path = r'C:\Users\Guna Shekar\Desktop\Dataset\unn.jpeg'
img = cv2.imread(img_path)
if img is not None:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_new = preprocess_signature_data([img_gray])
    label = clf.predict(X_new)[0]
    print('Prediction:', signature_classes[label])
else:
    print('Failed to read image:', img_path)
