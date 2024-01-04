from keras.models import load_model
import cv2
import numpy as np
from functions import ocr

class ImageClassifier:
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path, compile=False)
        self.class_names = open(labels_path, "r").readlines()

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1
        return img

    def predict(self, image_path):
        processed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(processed_image)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]
        return class_name[2:], confidence_score

def class_handler(class_name, imgPath):
    if class_name.rstrip('\n') == 'Judul_Lagu':
        ocrString = ocr.perform_ocr(imgPath)
        print(f"Img Text ==> {ocrString}")
    else:
        pass
    

def main():
    model_path = "model/keras_model.h5"
    labels_path = "model/labels.txt"

    classifier = ImageClassifier(model_path, labels_path)

    while True:
        image_path = input("Enter the path of the image (or 'exit' to end): ")
        
        if image_path.lower() == 'exit':
            break  # Exit the loop if the user enters 'exit'

        try:
            class_name, confidence_score = classifier.predict(image_path)
            output_string = f"Class: {class_name}Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
            print(output_string)
            class_handler(class_name, image_path)

        except Exception as e:
            print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
