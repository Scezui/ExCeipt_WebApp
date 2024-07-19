
# Dependencies
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, flash, send_from_directory, session, Response
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
from pathlib import Path
import platform
import pathlib
from pathlib import WindowsPath


# LLM
import argparse
from asyncio.log import logger
from Layoutlmv3_inference.ocr import prepare_batch_for_inference
from Layoutlmv3_inference.inference_handler import handle
import logging
import os
import copy


# Upload Folder
UPLOAD_FOLDER = r'static/temp/uploads'
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
        temp_folder = r'static/temp'
        inferenced_folder = r'static/temp/inferenced'

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
    UPLOAD_FOLDER = r'static/temp/uploads'
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
        if platform.system() == 'Windows':
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
            
            print(f"Prediction: {predictions}")

        return predictions

    except Exception as e:
        return {"error in make_predictions": str(e)}
    
    finally:
        if platform.system() == 'Windows':
            pathlib.WindowsPath = temp


@app.route('/predict/<filenames>', methods=['GET', 'POST'])
def predict_files(filenames):
    index_url = url_for('index')

    prediction_results = []
    image_paths = eval(filenames)  # Convert the filenames string back to a list
    
    for filename in image_paths:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        folder_path = UPLOAD_FOLDER
        destination_folder = r'static/temp/img_display'
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Get a list of all files in the source folder
        files = os.listdir(folder_path)

        # Loop through each file and copy it to the destination folder
        for file in files:
            # Construct the full path of the source file
            source_file_path = os.path.join(folder_path, file)
            
            # Construct the full path of the destination file
            destination_file_path = os.path.join(destination_folder, file)
            
            # Copy the file to the destination folder
            shutil.copy(source_file_path, destination_file_path)
        
        if os.path.exists(file_path):
            # Call make_predictions automatically
            prediction_result = make_predictions([file_path])
            if isinstance(prediction_result, list) and len(prediction_result) > 0:
                prediction_results.append(prediction_result[0])  # Append only the first prediction result
            else:
                print(f"Error making prediction for {file}: {prediction_result}")

            prediction_results_copy = copy.deepcopy(prediction_results)

            non_receipt_indices = []
            for i, prediction in enumerate(prediction_results):
                if prediction == 'non-receipt':
                    non_receipt_indices.append(i)

            # Delete images in reverse order to avoid index shifting
            for index in non_receipt_indices[::-1]:
                file_to_remove = os.path.join('static', 'temp', 'uploads', image_paths[index])
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)
                    
    return render_template('extractor.html', index_url=index_url, image_paths=image_paths, prediction_results = prediction_results, predictions=dict(zip(image_paths, prediction_results_copy)))
    

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
        images_path = r"static/temp/uploads/"
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
def replace_symbols_with_period(text):
    # Replace all non-alphanumeric characters with a period
    text = re.sub(r'\W+', '.', text)
    return text


@app.route('/create_csv', methods=['GET'])
def create_csv():
    try:
        # Path to the folder containing JSON files
        json_folder_path = r"static/temp/labeled"  # Change this to your folder path

        # Path to the output CSV folder
        output_folder_path = r"static/temp/inferenced/csv_files"
        os.makedirs(output_folder_path, exist_ok=True)

        column_order = [
            'RECEIPTNUMBER', 'MERCHANTNAME', 'MERCHANTADDRESS',
            'TRANSACTIONDATE', 'TRANSACTIONTIME', 'ITEMS',
            'PRICE', 'TOTAL', 'VATTAX'
        ]
#  Save
        # Iterate through JSON files in the folder
        for filename in os.listdir(json_folder_path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(json_folder_path, filename)

                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    all_data = data.get('output', [])

                # Initialize a dictionary to store labels and corresponding texts for this JSON file
                label_texts = {}
                for item in all_data:
                    label = item['label']
                    text = item['text'].replace('|', '')  # Strip the pipe character
                    if label == 'VATTAX' or label == 'TOTAL':
                        text = replace_symbols_with_period(text.replace(' ', ''))  # Remove spaces and replace symbols with periods
                    
                    if label == 'TRANSACTIONTIME':
                        # Concatenate all words for 'TRANSACTIONTIME' labels
                        if label in label_texts:
                            label_texts[label][0] += ": " + text  # Add a colon and a space before the text
                        else:
                            label_texts[label] = [text]
                    else:
                        if label in label_texts:
                            label_texts[label].append(text)
                        else:
                            label_texts[label] = [text]

                # Writing data to CSV file with ordered columns
                csv_file_path = os.path.join(output_folder_path, os.path.splitext(filename)[0] + '.csv')
                with open(csv_file_path, 'w', encoding='utf-8') as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=column_order, delimiter=",")
                    if os.path.getsize(csv_file_path) == 0:
                        csv_writer.writeheader()

                    # Constructing rows for the CSV file
                    num_items = len(label_texts.get('ITEMS', []))
                    for i in range(num_items):
                        row_data = {}
                        for label in column_order:
                            if label in label_texts:  # Check if the label exists in the dictionary
                                if label == 'ITEMS' or label == 'PRICE':
                                    if i < len(label_texts.get(label, [])):
                                        row_data[label] = label_texts[label][i]
                                    else:
                                        row_data[label] = ''
                                else:
                                    row_data[label] = label_texts[label][0]
                            else:
                                row_data[label] = ''  # If the label does not exist, set the value to an empty string
                        csv_writer.writerow(row_data)

            # Combining contents of CSV files into a single CSV file
        output_file_path = r"static/temp/inferenced/output.csv"
        with open(output_file_path, 'w', newline='', encoding='utf-8') as combined_csvfile:
            combined_csv_writer = csv.DictWriter(combined_csvfile, fieldnames=column_order, delimiter=",")
            combined_csv_writer.writeheader()

            # Iterate through CSV files in the folder
            for csv_filename in os.listdir(output_folder_path):
                if csv_filename.endswith(".csv"):
                    csv_file_path = os.path.join(output_folder_path, csv_filename)

                    # Read data from CSV file and write to the combined CSV file
                    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        for row in csv_reader:
                            combined_csv_writer.writerow(row)

        return '', 204  # Return an empty response with a 204 status code

    except Exception as e:
        print(f"An error occurred in create_csv: {str(e)}")
        return None

    except Exception as e:
        print(f"An error occurred in create_csv: {str(e)}")
        return None

    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        return jsonify({'error': 'File not found.'}), 404
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}")
        return jsonify({'error': 'JSON decoding error.'}), 500
    except csv.Error as e:
        print(f"CSV error: {str(e)}")
        return jsonify({'error': 'CSV error.'}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
        
@app.route('/get_data')
def get_data():
    return send_from_directory('static/temp/inferenced','output.csv', as_attachment=False)


@app.route('/download_csv', methods=['POST'])
def download_csv():
    try:
        csv_data = request.data.decode('utf-8')  # Get the CSV data from the request
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition":
                    "attachment; filename=output.csv"})
    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)