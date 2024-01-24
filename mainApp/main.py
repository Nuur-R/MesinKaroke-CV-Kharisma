from roboflow import Roboflow
import cv2
import numpy as np
import tensorflow as tf
from imageProcessor import process_image
from ocr import perform_ocr
import asyncio

API_KEY = "6zVobWAJvOh5A5fjkRot"
# VIDEO_PATH = "../test/Test3.mp4"  # From Video
VIDEO_PATH = 2 # From Camera

# Load model TFLite dan label
interpreter = tf.lite.Interpreter(model_path='models/model_unquant.tflite')
interpreter.allocate_tensors()

with open('models/labels.txt', 'r') as file:
    LABELS = file.read().splitlines()

IMAGE_PROCESSING_STATE = True

def image_processing():
    print('\n= = = = = Start Process Image = = = = =')
    process_image(API_KEY, 'images/OnList.jpg')
    print('\n= = = = = Start OCR = = = = =')
    ocr_string = perform_ocr('images/output_cropped.png')
    print(f'OCR String : {ocr_string}')
    global IMAGE_PROCESSING_STATE
    IMAGE_PROCESSING_STATE = True

# Buka video
cap = cv2.VideoCapture(VIDEO_PATH)
# Dapatkan resolusi asli sumber video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while cap.isOpened():
    # Baca frame
    ret, frame = cap.read()

    if not ret:
        print("Video selesai atau tidak dapat dibaca.")
        break

    # Preprocessing frame
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Menggunakan model TFLite
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Konversi input_frame ke FLOAT32
    input_frame = input_frame.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = LABELS[np.argmax(predictions)]

    is_on_list = True if predicted_label == '0 OnList' else False

    if is_on_list==True:
        IMAGE_PROCESSING_STATE = False
        cv2.imwrite('images/OnList.jpg', frame)
    elif is_on_list==False and IMAGE_PROCESSING_STATE==False:
        image_processing()

    # Tampilkan hasil di layar
    cv2.putText(frame, f'Label: {predicted_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Classification', frame)

    # Cek tombol keyboard untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup video dan jendela
cap.release()
cv2.destroyAllWindows()
