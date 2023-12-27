import pytesseract
from PIL import Image

def perform_ocr(image_path, tesseract_path=None):
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    try:
        # Open the image using Pillow (PIL)
        img = Image.open(image_path)

        # Perform OCR on the image
        text = pytesseract.image_to_string(img)

        return text
    except Exception as e:
        return f"Error performing OCR: {e}"