import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pytesseract
import easyocr
import requests
import urllib.parse
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/book_images/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

reader = easyocr.Reader(['en'])

# Helper function to process the uploaded image
def process_bookshelf(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600, 800))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sum_r = np.sum(image[:, :, 0], axis=0)
    sum_g = np.sum(image[:, :, 1], axis=0)
    sum_b = np.sum(image[:, :, 2], axis=0)
    combined_sum = (sum_r + sum_g + sum_b) / 3
    gray_sum = np.sum(gray, axis=0)
    peaks_gray, _ = find_peaks(-gray_sum, distance=30, prominence=500)
    peaks_color, _ = find_peaks(-combined_sum, distance=30, prominence=500)
    all_peaks = np.unique(np.concatenate((peaks_gray, peaks_color)))

    # Draw vertical lines
    output_image = image.copy()
    height, width = image.shape[:2]
    for peak in all_peaks:
        cv2.line(output_image, (peak, 0), (peak, height), (0, 255, 0), 2)

    # Save segmented book spines
    all_peaks_sorted = np.sort(all_peaks)
    all_peaks_sorted = np.concatenate(([0], all_peaks_sorted, [width]))
    book_images = []
    for i in range(len(all_peaks_sorted) - 1):
        left = all_peaks_sorted[i]
        right = all_peaks_sorted[i + 1]
        book_spine = image[:, left:right]
        book_images.append(book_spine)
        cv2.imwrite(f"{app.config['RESULT_FOLDER']}/book_spine_{i + 1}.jpg", book_spine)

    return book_images, len(book_images)

# Helper function to perform OCR and search Google Books
def search_books(book_images):
    book_data = []
    for i, book_spine in enumerate(book_images):
        rotated_spine = cv2.rotate(book_spine, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb_spine = cv2.cvtColor(rotated_spine, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb_spine)
        ocr_text = ' '.join([result[1] for result in results]).strip()

        if ocr_text:
            encoded_query = urllib.parse.quote(ocr_text)
            url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_query}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'items' in data:
                    book = data['items'][0]['volumeInfo']
                    title = book.get('title', 'No Title Found')
                    authors = ', '.join(book.get('authors', ['No Author Found']))
                    book_data.append({'Spine': i + 1, 'OCR': ocr_text, 'Title': title, 'Authors': authors})
        else:
            book_data.append({'Spine': i + 1, 'OCR': 'No Text Found', 'Title': 'N/A', 'Authors': 'N/A'})

    return book_data

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Save uploaded file
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process the image
            book_images, count = process_bookshelf(filepath)
            book_data = search_books(book_images)

            # Save data to CSV
            df = pd.DataFrame(book_data)
            df.to_csv(os.path.join(app.config['RESULT_FOLDER'], 'book_spine_results.csv'), index=False)

            return render_template('results.html', book_data=book_data, count=count)
    return render_template('upload.html')

@app.route('/results/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
