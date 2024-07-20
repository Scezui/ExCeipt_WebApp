# Synopsis
This project aims to automate bookkeeping by developing a web application that extracts information from receipts. The system scans an image to determine if it is a receipt. If confirmed, it uses the LayoutLMv3 model to automatically extract and categorize the text. The extracted information is then saved to a CSV file for further analysis and record-keeping.

# Code Example
### Import to Export

1. **Upload Files**: Users can upload multiple receipt images.
2. **Make Predictions**: The application processes the images, makes predictions, and filters non-receipt images.
3. **Run Inference**: Process the images to extract text and categorize it.
4. **Create CSV**: Save the extracted information into a CSV file.

```python
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory, Response
import os
import shutil
import json
import csv
from werkzeug.utils import secure_filename
from fastai.vision import load_learner, open_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/temp/uploads'
app.config['SECRET_KEY'] = 'supersecretkey'

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load the index function

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files[]')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('predict_files', filenames=filenames))

def make_predictions(image_paths):
    learner = load_learner('model/export')
    predictions = []
    for image_path in image_paths:
        image = open_image(image_path)
        prediction_class, _, _ = learner.predict(image)
        predictions.append(str(prediction_class))
    return predictions

@app.route('/run_inference', methods=['GET'])
def run_inference():
    process_images('model', 'static/temp/uploads')
    return redirect(url_for('create_csv'))

def process_images(model_path, images_path):
    inference_batch = prepare_batch_for_inference([os.path.join(images_path, f) for f in os.listdir(images_path)])
    handle(inference_batch, {"model_dir": model_path})

# Other routes and functions...

@app.route('/create_csv', methods=['GET'])
def create_csv():
  # Function to export to CSV

@app.route('/get_data')
def get_data():
    return send_from_directory('static/temp/inferenced', 'output.csv', as_attachment=False)

if __name__ == '__main__':
    app.run()
```

# Motivation
Efficient bookkeeping is essential for business 
operations, but manually encoding receipts can be time-consuming and error-prone. This study develops an automated 
system to extract structured data from receipts using 
machine learning, streamlining the bookkeeping process. 

# Installation
### Prerequisites
- Git: Ensure you have Git installed on your machine. You can download it from [here.](https://git-scm.com/downloads).
- Git LFS: Ensure Git LFS (Large File Storage) is installed. You can download it from [here.](https://git-lfs.com/).
- Python 3.11.4: Make sure you have Python 3.11.4 installed. You can download it from [here.](https://www.python.org/downloads/release/python-3114/).
- Conda: If you prefer using Conda for environment management, ensure it is installed. You can download it from [here.](https://www.anaconda.com/download).
- OCR.Space API key: Create your free API key [here.](https://ocr.space/ocrapi)

### Steps to Set Up the Project
1. Clone the Repository
    ```
      git clone https://github.com/Scezui/ExCeipt_WebApp.git
      cd ExCeipt_Webapp
    ```
2. Pull LFS files
    ```
      git lfs pull
    ```
3. Create a new environment

    Using _conda_:
    ```
    conda create --name myenv python=3.11.4
    conda activate myenv
    ```
    Using _venv_:
    ```
    python -m venv myenv
    source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
    ```
4. Navigate to the source directory
    ```
    cd src
    ```
5. Install Dependencies
    ```
    pip install -r requirements.txt
    ```
6. Add API key
   
    Enter the API key you created from OCR.Space
    ```
    python create_api.py
    ```
8. Run the web application
    ```
    flask run
    ```
OR you can access a working web application [here.](https://innovex-exceipt.hf.space)


# API Reference
This project leverages the OCR.Space API to perform Optical Character Recognition (OCR) on images. This enables the extraction of text from images in various formats and languages.

For detailed information on the API endpoints, parameters, response formats and getting a free API key, refer to the [OCR.Space](https://ocr.space/ocrapi) API documentation.

# Contributors | Group Members
- _Fagarita, Dave F._
- _Servandil, Jimuel S._
- _Magno, Jannica Mae G._
- _Catanus, Jeziah Lois C._
