import imutils
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet
import os
from google.colab.patches import cv2_imshow
# Initialize FaceNet for feature extraction
embedder = FaceNet()

# Face detection model paths
prototxt = 'deploy.prototxt.txt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Function to detect and extract face embeddings
def extract_features(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            face = cv2.resize(face, (160, 160))
            face = np.expand_dims(face, axis=0)
            
            # Extract embeddings using FaceNet
            embedding = embedder.embeddings(face)[0]
            return embedding, (startX, startY, endX, endY)
    return None, None

# Load images for comparison
def load_images(directory):
    embeddings = []
    labels = []
    for filename in os.listdir(directory):
        label = filename.split("_")[0]
        image = cv2.imread(os.path.join(directory, filename))
        embedding, _ = extract_features(image)
        if embedding is not None:
            embeddings.append(embedding)
            labels.append(label)
    return np.array(embeddings), np.array(labels)

# Recognize face based on similarity
def recognize_face(face_embedding, known_embeddings, known_labels):
    similarities = cosine_similarity([face_embedding], known_embeddings)
    best_match_index = np.argmax(similarities)
    return known_labels[best_match_index], similarities[0][best_match_index]

# Evaluation metrics
def evaluate_accuracy(known_embeddings, known_labels):
    correct = 0
    total = len(known_labels)
    for i, embedding in enumerate(known_embeddings):
        predicted_label, _ = recognize_face(embedding, known_embeddings, known_labels)
        if predicted_label == known_labels[i]:
            correct += 1
    accuracy = correct / total
    print(f"Recognition Accuracy: {accuracy * 100:.2f}%")

# Load known images
known_embeddings, known_labels = load_images('known_faces/')

# Perform face detection and recognition on input image
image_file = 'photof.jpg'
image = cv2.imread(image_file)
image = imutils.resize(image, width=400)
embedding, box = extract_features(image)

if embedding is not None:
    print("[INFO] Face Detected and Features Extracted")
    predicted_label, similarity = recognize_face(embedding, known_embeddings, known_labels)
    print(f"Recognized as: {predicted_label} with similarity: {similarity:.2f}")
    evaluate_accuracy(known_embeddings, known_labels)
    (startX, startY, endX, endY) = box
    text = f"{predicted_label} ({similarity * 100:.2f}%)"
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2_imshow(image)

else:
    print("[INFO] No Face Detected")
