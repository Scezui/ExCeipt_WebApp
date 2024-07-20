# Synopsis
This project aims to automate bookkeeping by developing a web application that extracts information from receipts. The system scans an image to determine if it is a receipt. If confirmed, it uses the LayoutLMv3 model to automatically extract and categorize the text. The extracted information is then saved to a CSV file for further analysis and record-keeping.

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
6. Install Dependencies
    ```
    pip install -r requirements.txt
    ```
6. Run the web application
    ```
    flask run
    ```
OR you can access a working web application [here.](innovex-exceipt.hf.space)


# API Reference
This project leverages the OCR.Space API to perform Optical Character Recognition (OCR) on images. This enables the extraction of text from images in various formats and languages.

For detailed information on the API endpoints, parameters, response formats and getting a free API key, refer to the [OCR.Space](https://ocr.space/ocrapi) API documentation.

# Contributors | Group Members
- _Fagarita, Dave F._
- _Servandil, Jimuel S._
- _Magno, Jannica Mae G._
- _Catanus, Jeziah Lois C._
