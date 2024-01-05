from roboflow import Roboflow
import json

rf = Roboflow(api_key="6zVobWAJvOh5A5fjkRot")
project = rf.workspace().project("mesin-karoke")
model = project.version(2).model

# infer on a local image
imageJson =model.predict("test/test6.png", confidence=40, overlap=30).json()
print(imageJson)
print(type(imageJson))
# save JSON
with open('data.json', 'w') as file:
    json.dump(imageJson, file)

print("Data berhasil disimpan ke data.json")

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())