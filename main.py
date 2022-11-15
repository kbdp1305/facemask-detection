import os
import numpy as np

from PIL import Image

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import cv2
import mahotas
import pickle
import cv2


def get_image(name, folder):
    filepath = os.path.join(folder, name)
    img = Image.open(filepath)
    return np.array(img)


def get_images(directoryName):
    directory = os.fsencode(directoryName)
    images = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            image = get_image(filename, directoryName)
            images.append(image)
            continue
        else:
            continue
    return images


def fd_hu_moments(image):
    # Compute the Hu Moments of the image as a feature
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    # Compute the haralick texture feature vector from the image
    haralick = mahotas.features.haralick(image).mean(axis=0)
    return haralick


def fd_histogram(image):
    # Compute the histogram of the image as a feature
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    hist = hist.flatten()
    return hist


def getFeatures(image):
    return np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])


if __name__ == '__main__':
    # Get features for the training images with masks
    print("Processing training mask images")
    mask_images = get_images("images/mask_train")

    mask_features = []
    for image in mask_images:
        mask_features.append(getFeatures(image))

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Normalize The feature vectors...
    mask_features = scaler.fit_transform(mask_features)

    # Save features to a file
    with open('mask_features.pkl', 'wb') as f:
        pickle.dump(mask_features, f)

    print("Training Mask Features has shape:", np.shape(mask_features))

    # Get features for the training images without masks
    print("Processing training no mask images")
    nomask_images = get_images("images/unmask_train")

    nomask_features = []
    for image in nomask_images:
        nomask_features.append(getFeatures(image))

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Normalize The feature vectors...
    nomask_features = scaler.fit_transform(nomask_features)

    # Save features to a file
    with open('nomask_features.pkl', 'wb') as f:
        pickle.dump(nomask_features, f)

    print("Training No Mask Features has shape:", np.shape(nomask_features))

    # Get features for test images with mask
    print("Processing test mask images")
    mask_images_test = get_images("images/mask_test")

    mask_features_test = []
    for image in mask_images_test:
        mask_features_test.append(getFeatures(image))

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Normalize The feature vectors...
    mask_features_test = scaler.fit_transform(mask_features_test)

    # Save features to a file
    with open('mask_features_test.pkl', 'wb') as f:
        pickle.dump(mask_features_test, f)
    print("Testing Mask Features has shape:", np.shape(mask_features_test))

    # Get features for test images without mask
    print("Processing test no mask images")
    nomask_images_test = get_images("images/unmask_test")

    nomask_features_test = []
    for image in nomask_images_test:
        nomask_features_test.append(getFeatures(image))

    print(nomask_features_test)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Normalize The feature vectors...
    nomask_features_test = scaler.fit_transform(nomask_features_test)

    with open('nomask_features_test.pkl', 'wb') as f:
        pickle.dump(nomask_features_test, f)

    print("Testing No Mask Features has shape:",
          np.shape(nomask_features_test))

    # Read feature data from file
    # with open('mask_features.pkl', 'rb') as f:
    #     mask_features = pickle.load(f)
    #
    # with open('nomask_features.pkl', 'rb') as f:
    #     nomask_features = pickle.load(f)
    #
    # with open('mask_features_test.pkl', 'rb') as f:
    #     mask_features_test = pickle.load(f)
    #
    # with open('nomask_features_test.pkl', 'rb') as f:
    #     nomask_features_test = pickle.load(f)

    # Image labels. Mask = 1.0, No Mask = 0.0
    mask_labels = [1.0 for i in mask_features]
    nomask_labels = [0.0 for i in nomask_features]
    mask_labels_test = [1.0 for i in mask_features_test]
    nomask_labels_test = [0.0 for i in mask_features_test]

    # Combine mask/no mask features for train and test set
    X_train = np.concatenate([mask_features, nomask_features])
    y_train = np.concatenate([mask_labels, nomask_labels])

    X_test = np.concatenate([mask_features_test, nomask_features_test])
    y_test = np.concatenate([mask_labels_test, nomask_labels_test])

    # Train the SVM Model
    print("Training...")
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model accuracy is: ', accuracy)

    # It helps in identifying the faces
    import cv2
    import sys
    import numpy
    import os
    size = 4
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'datasets'

    # Part 1: Create fisherRecognizer
    print('Recognizing Face Please Be in sufficient Lights...')

    # Create a list of images and a list of corresponding names
    # (images, labels, names, id) = ([], [], {}, 0)
    # for (subdirs, dirs, files) in os.walk(datasets):
    #     for subdir in dirs:
    #         names[id] = subdir
    #         subjectpath = os.path.join(datasets, subdir)
    #         for filename in os.listdir(subjectpath):
    #             path = subjectpath + '/' + filename
    #             label = id
    #             images.append(cv2.imread(path, 0))
    #             labels.append(int(label))
    #         id += 1
    (width, height) = (130, 100)

    # # Create a Numpy array from the two lists above
    # (images, labels) = [numpy.array(lis) for lis in [images, labels]]

    # # OpenCV trains a model from the images
    # # NOTE FOR OpenCV2: remove '.face'
    # model = cv2.face.LBPHFaceRecognizer_create()
    # model.train(images, labels)

    # Part 2: Use fisherRecognizer on camera stream
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))

            input_feature = np.array(getFeatures(
                face_resize))
            input_features = np.array(input_feature)
            scaler = MinMaxScaler(feature_range=(0, 1))
            input_features = scaler.fit_transform(input_features.reshape(1, -1))
            print(input_features)
            y_pred = svm.predict(input_features)
            # Try to recognize the face
            # prediction = svm.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            cv2.imshow('OpenCV', im)

            key = cv2.waitKey(10)
            if key == 27:
                break
