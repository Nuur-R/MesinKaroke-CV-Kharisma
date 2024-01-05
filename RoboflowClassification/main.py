from roboflow import Roboflow
import cv2
import json

inputImage = "../test/test6.png"

def crop_image(input_path, output_path, coordinates):
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

rf = Roboflow(api_key="6zVobWAJvOh5A5fjkRot")
project = rf.workspace().project("mesin-karoke")
model = project.version(2).model

# infer on a local image
model.predict(inputImage, confidence=40, overlap=30).save("prediction.jpg")
imageJson =model.predict(inputImage, confidence=40, overlap=30).json()
print(imageJson)
print(type(imageJson))



# Menampilkan informasi
print("\n = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = \n")
predictions = imageJson["predictions"]
image_dimensions = imageJson["image"]
print("Dimensi Gambar:", image_dimensions["width"], "x", image_dimensions["height"])
print("\nPrediksi Objek:")
for prediction in predictions:
    print(f"\nClass: {prediction['class']}")
    print(f"Koordinat: x={prediction['x']}, y={prediction['y']}, width={prediction['width']}, height={prediction['height']}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Path Gambar: {prediction['image_path']}")

print("\n = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = \n")
judul_musik_hover_data = [
    prediction for prediction in imageJson.get('predictions', [])
    if prediction.get('class') == 'Judul Musik Hover'
]

# Menampilkan data "Judul Musik Hover" dengan nilai confidence tertinggi
judul_musik_hover_data = [
    prediction for prediction in imageJson.get('predictions', [])
    if prediction.get('class') == 'Judul Musik Hover'
]

# Menampilkan data "Judul Musik Hover" dengan nilai confidence tertinggi dan melakukan cropping
if judul_musik_hover_data:
    # Mengurutkan data berdasarkan nilai confidence secara menurun
    sorted_data = sorted(judul_musik_hover_data, key=lambda x: x['confidence'], reverse=True)
    
    # Mengambil data dengan nilai confidence tertinggi
    highest_confidence_data = sorted_data[0]
    
    print("Class:", highest_confidence_data['class'])
    print("Koordinat:", f"x={highest_confidence_data['x']}, y={highest_confidence_data['y']}, width={highest_confidence_data['width']}, height={highest_confidence_data['height']}")
    print("Confidence:", highest_confidence_data['confidence'])
    print("Path Gambar:", highest_confidence_data['image_path'])
    print("Prediction Type:", highest_confidence_data['prediction_type'])
    
    # Memanggil fungsi crop untuk mencrop gambar
    crop_image(highest_confidence_data['image_path'], "output_cropped.jpg", highest_confidence_data)
else:
    print("Tidak ada data dengan class 'Judul Musik Hover' dalam file JSON.")
