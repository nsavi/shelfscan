import os
import base64
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import easyocr
from scipy.signal import find_peaks
from werkzeug.utils import secure_filename
import urllib.parse
import requests

app = Flask(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Image processing route
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            books = process_books(file_path)
            return jsonify({'books': books})

        except Exception as e:
            return jsonify({"error": f"Error processing image: {e}"}), 500

    return jsonify({"error": "Invalid file type"}), 400

# Function to process the uploaded image (detect book spines, perform OCR)
def process_books(file_path):
    try:
        image = cv2.imread(file_path)
        if image is None:
            return []

        # Resize for easier processing
        image = cv2.resize(image, (600, 800))
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the sum of pixel intensities for each column in RGB channels
        sum_r = np.sum(image[:, :, 0], axis=0)  # Red channel
        sum_g = np.sum(image[:, :, 1], axis=0)  # Green channel
        sum_b = np.sum(image[:, :, 2], axis=0)  # Blue channel

        # Combine the RGB sums to detect color transitions
        combined_sum = (sum_r + sum_g + sum_b) / 3

        # Detect peaks (book boundaries) using grayscale and color information
        gray_sum = np.sum(gray, axis=0)
        peaks_gray, _ = find_peaks(-gray_sum, distance=30, prominence=500)
        peaks_color, _ = find_peaks(-combined_sum, distance=30, prominence=500)

        all_peaks = np.unique(np.concatenate((peaks_gray, peaks_color)))

        # Sort peaks and add start/end points to cover the first and last books
        all_peaks_sorted = np.sort(all_peaks)
        width = image.shape[1]
        all_peaks_sorted = np.concatenate(([0], all_peaks_sorted, [width]))

        # Segment each book and rotate them
        book_images = []
        for i in range(len(all_peaks_sorted) - 1):
            left = all_peaks_sorted[i]
            right = all_peaks_sorted[i + 1]
            book_spine = image[:, left:right]
            rotated_spine = cv2.rotate(book_spine, cv2.ROTATE_90_COUNTERCLOCKWISE)
            book_images.append(rotated_spine)

        # Perform OCR on each rotated book spine
        ocr_results = []
        for book_spine in book_images:
            result = reader.readtext(book_spine)
            ocr_text = ' '.join([text[1] for text in result])
            ocr_results.append(ocr_text)

        # Prepare the results for the frontend
        output_books = []
        for i, book in enumerate(book_images):
            _, buffer = cv2.imencode('.jpg', book)
            img_str = buffer.tobytes()
            img_b64 = f"data:image/jpeg;base64,{base64.b64encode(img_str).decode()}"
            book_info = {
                'image': img_b64,
                'ocrText': ocr_results[i],
            }

            # Search Google Books API for more information
            book_info['googleBooks'] = search_google_books(ocr_results[i])

            output_books.append(book_info)

        return output_books

    except Exception as e:
        return []

# Function to search Google Books API for book details
def search_google_books(query):
    """Searches the Google Books API and returns title, authors, description, and image URL."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_query}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        if 'items' in data:
            book = data['items'][0]['volumeInfo']
            title = book.get('title', 'No Title Found')
            authors = ', '.join(book.get('authors', ['No Author Found']))
            description = book.get('description', 'No Description Available')
            image_url = book.get('imageLinks', {}).get('thumbnail', 'No Image Available')
            return {
                'title': title,
                'authors': authors,
                'description': description,
                'imageUrl': image_url
            }
    return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
