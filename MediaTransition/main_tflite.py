import cv2
import numpy as np
import tensorflow as tf

# Load model TFLite dan label
interpreter = tf.lite.Interpreter(model_path='tflite/model_unquant.tflite')
interpreter.allocate_tensors()

with open('tflite/labels.txt', 'r') as file:
    labels = file.read().splitlines()

# Buka video
video_path =  '../test/Test2.mp4'  # Ganti dengan path video kamu
cap = cv2.VideoCapture(video_path)

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
    predicted_label = labels[np.argmax(predictions)]
    
    if predicted_label == '0 OnList':
        cv2.imwrite('OnList.jpg', frame)
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
