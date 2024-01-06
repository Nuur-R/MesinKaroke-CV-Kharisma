import cv2
import numpy as np
from keras.models import load_model

# Load model dan label
model = load_model('model/keras_model.h5')
with open('model/labels.txt', 'r') as file:
    labels = file.read().splitlines()

# Buka video
video_path = '../test/Test2.mp4'  # Ganti dengan path video kamu
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

    # Prediksi kelas
    predictions = model.predict(input_frame)
    predicted_label = labels[np.argmax(predictions)]

    # Tampilkan hasil di layar
    cv2.putText(frame, f'Label: {predicted_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Classification', frame)

    # Simpan gambar berdasarkan kondisi
    if predicted_label == '0 OnList':
        cv2.imwrite('onList.png', frame)

    # Cek tombol keyboard untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup video dan jendela
cap.release()
cv2.destroyAllWindows()
