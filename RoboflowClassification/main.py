from roboflow import Roboflow
import json
import cv2
from functions import ocr

class ImageProcessor:
    def __init__(self, api_key):
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace().project("mesin-karoke")
        self.model = self.project.version(2).model

    def crop_image(self, input_path, output_path, coordinates):
        try:
            # Buka gambar
            image = cv2.imread(input_path)

            # Hitung koordinat baru berdasarkan penyesuaian
            width = coordinates["width"]
            height = coordinates["height"]
            x = coordinates["x"] - (width / 2)
            y = coordinates["y"] - (height / 2)

            # Pastikan koordinat tidak keluar dari batas gambar
            x = max(0, x)
            y = max(0, y)

            # Lakukan cropping
            cropped_image = image[int(y):int(y + height), int(x):int(x + width)]

            # Simpan hasil cropping
            cv2.imwrite(output_path, cropped_image)

            print(f"Gambar berhasil di-crop dan disimpan sebagai {output_path}")
        except Exception as e:
            print(f"Terjadi kesalahan saat melakukan cropping: {str(e)}")

    def process_image(self, input_image_path):
        # infer on a local image
        model_predictions = self.model.predict(input_image_path, confidence=40, overlap=30)
        model_predictions.save("images/prediction.jpg")
        image_json = model_predictions.json()

        # Menampilkan informasi
        print("\nDimensi Gambar:", image_json["image"]["width"], "x", image_json["image"]["height"])
        print("\nPrediksi Objek:")

        predictions = image_json.get("predictions", [])
        if not predictions:
            print("Tidak ada hasil prediksi objek dalam gambar.")
            return

        for prediction in predictions:
            print(f"\nClass: {prediction['class']}")
            print(f"Koordinat: x={prediction['x']}, y={prediction['y']}, width={prediction['width']}, height={prediction['height']}")
            print(f"Confidence: {prediction['confidence']}")
            print(f"Path Gambar: {prediction['image_path']}")

        print("\n")

        # Menampilkan data "Judul Musik Hover" dengan nilai confidence tertinggi dan melakukan cropping
        judul_musik_hover_data = [
            prediction for prediction in predictions
            if prediction.get('class') == 'Judul Musik Hover'
        ]

        if judul_musik_hover_data:
            # Mengurutkan data berdasarkan nilai confidence secara menurun
            sorted_data = sorted(judul_musik_hover_data, key=lambda x: x['confidence'], reverse=True)

            # Mengambil data dengan nilai confidence tertinggi
            highest_confidence_data = sorted_data[0]

            print("Class:", highest_confidence_data['class'])
            print(f"Koordinat: x={highest_confidence_data['x']}, y={highest_confidence_data['y']}, width={highest_confidence_data['width']}, height={highest_confidence_data['height']}")
            print(f"Confidence: {highest_confidence_data['confidence']}")
            print(f"Path Gambar: {highest_confidence_data['image_path']}")
            print(f"Prediction Type: {highest_confidence_data['prediction_type']}")

            # Memanggil fungsi crop untuk mencrop gambar
            self.crop_image(highest_confidence_data['image_path'], "images/output_cropped.jpg", highest_confidence_data)
        else:
            print("Tidak ada data dengan class 'Judul Musik Hover' dalam hasil prediksi.")

if __name__ == "__main__":
    # Gunakan class ImageProcessor
    api_key = "6zVobWAJvOh5A5fjkRot"
    image_processor = ImageProcessor(api_key)

    while True:
        # Meminta input dari pengguna untuk path gambar
        input_image_path = input("Masukkan path gambar atau ketik 'exit' untuk keluar: ")

        # Cek apakah pengguna ingin keluar dari loop
        if input_image_path.lower() == 'exit':
            break

        # Memproses gambar
        image_processor.process_image(input_image_path)

        # Melakukan OCR pada gambar yang sudah di-crop
        ocrText = ocr.perform_ocr("images/output_cropped.jpg")
        print("Hasil OCR:", ocrText)