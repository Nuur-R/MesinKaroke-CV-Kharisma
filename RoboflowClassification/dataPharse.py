import json

# Baca data dari file JSON
with open('data.json', 'r') as file:
    data = json.load(file)

# Akses nilai-nilai dalam data
predictions = data['predictions']
image_info = data['image']

# Akses dimensi gambar
image_width = int(image_info['width'])
image_height = int(image_info['height'])

# Iterasi dan akses nilai-nilai dalam setiap prediksi
for prediction in predictions:
    x = prediction['x']
    y = prediction['y']
    width = prediction['width']
    height = prediction['height']
    confidence = prediction['confidence']
    class_name = prediction['class']
    class_id = prediction['class_id']
    image_path = prediction['image_path']
    prediction_type = prediction['prediction_type']

    # Lakukan sesuatu dengan nilai-nilai ini, misalnya, cetak atau simpan dalam struktur data lainnya
    print(f"Prediksi: {class_name}, Confidence: {confidence}, Koordinat: ({x}, {y}), Dimensi: {width}x{height}")
