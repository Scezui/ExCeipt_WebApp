from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, flash, send_from_directory, session
from PIL import Image, ImageDraw
import torch
from transformers import LayoutLMv2ForTokenClassification, LayoutLMv3Tokenizer
import csv
import json
import subprocess
import os
import torch
import warnings
from PIL import Image
import sys
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
from werkzeug.utils import secure_filename
import pandas as pd
from itertools import zip_longest
import inspect
from threading import Lock
import signal
import shutil
from datetime import datetime
import zipfile
# LLM
import argparse
from asyncio.log import logger
from Layoutlmv3_inference.ocr import prepare_batch_for_inference
from Layoutlmv3_inference.inference_handler import handle
import logging
import os
import warnings

# Ignore SourceChangeWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=SourceChangeWarning)


UPLOAD_FOLDER = 'temp/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'



# Added "temp" files cleaning for privacy and file managements.
# All temporary files were moved to "output_folders" for review and recovery.
# Moving of temp files were called at home page to ensure that new data were being supplied for extractor.
@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # Current date and time
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")

        # Source folders
        temp_folder = r'temp'
        inferenced_folder = r'inferenced'

        # Destination folder path
        destination_folder = os.path.join('output_folders', dt_string)  # Create a new folder with timestamp

        # Move the temp and inferenced folders to the destination folder
        shutil.move(temp_folder, destination_folder)
        shutil.move(inferenced_folder, destination_folder)

        return render_template('index.html', destination_folder=destination_folder)
    except:
        return render_template('index.html')
    

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    UPLOAD_FOLDER = 'temp/uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if request.method == 'POST':
        if 'files[]' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        files = request.files.getlist('files[]')
        filenames = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filenames.append(filename)
        return redirect(url_for('predict_files', filenames=filenames))
    return render_template('index.html')


def make_predictions(image_paths):
    try:
        # For Windows OS
        temp = pathlib.PosixPath  # Save the original state
        pathlib.PosixPath = pathlib.WindowsPath  # Change to WindowsPath temporarily
        
        model_path = Path(r'model/export')
        learner = load_learner(model_path)
        
        predictions = []

        for image_path in image_paths:
            # Open the image using fastai's open_image function
            image = open_image(image_path)

            # Make a prediction
            prediction_class, prediction_idx, probabilities = learner.predict(image)

            # If you want the predicted class as a string
            predicted_class_str = str(prediction_class)
            
            predictions.append(predicted_class_str)

        return predictions

    except Exception as e:
        return {"error": str(e)}
        
    finally:
        pathlib.PosixPath = temp 

@app.route('/predict/<filenames>', methods=['GET', 'POST'])
def predict_files(filenames):
    prediction_results = []
    image_paths = eval(filenames)  # Convert the filenames string back to a list
    
    for filename in image_paths:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            # Call make_predictions automatically
            prediction_result = make_predictions([file_path])  # Pass file_path as a list
            prediction_results.extend(prediction_result)  # Use extend to add elements of list to another list
            print(image_paths)
    
    return render_template('extractor.html', image_paths=image_paths, predictions=dict(zip(image_paths, prediction_results)))

    
    
@app.route('/get_inference_image')
def get_inference_image():
    # Assuming the new image is stored in the 'inferenced' folder with the name 'temp_inference.jpg'
    inferenced_image = 'inferenced/temp_inference.jpg'
    return jsonify(updatedImagePath=inferenced_image), 200  # Return the image path with a 200 status code
    

def process_images(model_path: str, images_path: str) -> None:
    try:
        image_files = os.listdir(images_path)
        images_path = [os.path.join(images_path, image_file) for image_file in image_files]
        inference_batch = prepare_batch_for_inference(images_path)
        context = {"model_dir": model_path}
        handle(inference_batch, context)
    except Exception as err:
        os.makedirs('log', exist_ok=True)
        logging.basicConfig(filename='log/error_output.log', level=logging.ERROR,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')
        logger = logging.getLogger(__name__)
        logger.error(err)

@app.route('/run_inference', methods=['GET'])
def run_inference():
    try:
        model_path = r"model"
        images_path = r"temp/uploads/"
        process_images(model_path, images_path)
        return redirect(url_for('create_csv'))
    except Exception as err:
        return f"Error processing images: {str(err)}", 500


@app.route('/stop_inference', methods=['GET'])
def stop_inference():
    try:
        # Get the process ID of the run_inference process
        run_inference_pid = os.getpid()  # Assuming it's running in the same process

        # Send the SIGTERM signal to gracefully terminate the process
        os.kill(run_inference_pid, signal.SIGTERM)

        return render_template('index.html')
    except ProcessLookupError:
        logging.warning("run_inference process not found.")
    except Exception as err:
        logging.error(f"Error terminating run_inference process: {err}")

# Define a function to replace all symbols with periods
def replace_symbols_with_period(value):
    # return re.sub(r'\W+', '.', str(value))
    return value.replace(',', '.')


from itertools import zip_longest

@app.route('/create_csv', methods=['GET'])
def create_csv():
    try:
        # Path to the folder containing JSON files
        json_folder_path = r"temp/labeled"  # Change this to your folder path
        
        # Path to the output CSV folder
        output_folder_path = r"inferenced/csv_files"
        os.makedirs(output_folder_path, exist_ok=True)

        # Initialize an empty list to store all JSON data
        all_data = []

        # Iterate through JSON files in the folder
        for filename in os.listdir(json_folder_path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(json_folder_path, filename)

                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                    all_data.extend(data['output'])

                # Creating a dictionary to store labels and corresponding texts for this JSON file
                label_texts = {}
                for item in data['output']:
                    label = item['label']
                    text = item['text']
                        
                    if label not in label_texts:
                        label_texts[label] = []
                    label_texts[label].append(text)

                # Order of columns as requested
                column_order = [
                    'RECEIPTNUMBER', 'MERCHANTNAME', 'MERCHANTADDRESS', 
                    'TRANSACTIONDATE', 'TRANSACTIONTIME', 'ITEMS', 
                    'PRICE', 'TOTAL', 'VATTAX'
                ]

                # Writing data to CSV file with ordered columns
                csv_file_path = os.path.join(output_folder_path, os.path.splitext(filename)[0] + '.csv')
                with open(csv_file_path, 'w', newline='') as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=column_order, delimiter=",")
                    csv_writer.writeheader()

                    # Iterate through items and prices
                    max_length = max(len(label_texts.get('ITEMS', [])), len(label_texts.get('PRICE', [])))
                    for i in range(max_length):
                        # Prepare data for each row
                        items = label_texts.get('ITEMS', [])[i] if i < len(label_texts.get('ITEMS', [])) else ''
                        prices = label_texts.get('PRICE', [])[i] if i < len(label_texts.get('PRICE', [])) else ''

                        # Check if items and prices are separated by space
                        if ' ' in items or ' ' in prices:
                            item_list = items.split() if items else []
                            price_list = prices.split() if prices else []

                            # Create new rows for each combination of items and prices
                            for item, price in zip(item_list, price_list):
                                row_data = {label: replace_symbols_with_period(label_texts[label][i]) if label == 'ITEMS' else replace_symbols_with_period(label_texts[label][i]) for label in column_order}
                                row_data['ITEMS'] = item
                                row_data['PRICE'] = price
                                csv_writer.writerow(row_data)
                        else:
                            # Write the row to CSV
                            row_data = {label: replace_symbols_with_period(label_texts[label][i]) if i < len(label_texts[label]) else '' for label in column_order}
                            csv_writer.writerow(row_data)

        # Combining contents of CSV files into a single CSV file
        output_file_path = r"inferenced/output.csv"
        with open(output_file_path, 'w', newline='') as combined_csvfile:
            combined_csv_writer = csv.DictWriter(combined_csvfile, fieldnames=column_order, delimiter=",")
            combined_csv_writer.writeheader()

            # Iterate through CSV files in the folder
            for csv_filename in os.listdir(output_folder_path):
                if csv_filename.endswith(".csv"):
                    csv_file_path = os.path.join(output_folder_path, csv_filename)

                    # Read data from CSV file and write to the combined CSV file
                    with open(csv_file_path, 'r') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        for row in csv_reader:
                            combined_csv_writer.writerow(row)

        return '', 204  # Return an empty response with a 204 status code

    except Exception as e:
        # Handle exceptions here
        return str(e), 500  # Return error message with a 500 status code if an exception occurs
        
@app.route('/get_data')
def get_data():
    return send_from_directory('inferenced','output.csv', as_attachment=False)

from flask import jsonify

@app.route('/download_csv', methods=['GET'])
def download_csv():
    try:
        output_file_path = r"inferenced/output.csv"  # path to output CSV file
        # Check if the file exists
        if os.path.exists(output_file_path):
            return send_file(output_file_path, as_attachment=True, download_name='output.csv')
        else:
            return jsonify({"error": "CSV file not found"})
    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)}"})



if __name__ == '__main__':
    app.run(debug=True)