# Page-Stream-Segmentation-of-Business-Documents
Bachelor thesis project developing a PSS System with a multi-modal neural network. 

## Run API
The API is available as a docker container. Docker has to be installed to use the container.
Run the following commands to start the API.

    docker build -t pss-business-documents .
    docker run -d -p 80:80 --name pss-api pss-business-documents

To lower the batch size used for inference set the environment variable **BATCH_SIZE**.

## Google Colab 
The API is also available as a Google Colab notebook.
Open the following link and follow the instructions in the notebook to run the API: 

https://colab.research.google.com/github/Qrauli/Page-Stream-Segmentation-of-Business-Documents/blob/main/PSS_Business_Documents.ipynb

## Training and Testing
A sufficiently powerful Nvidia graphics card is needed to retrain the model.

To retrain the model run the **training.py** script in the **models** folder.
For testing the model use the **evaluate.py** script.

The dataset can be found at https://github.com/aldolipani/TABME in the **data** folder.
Run the **download_dataset.sh** file to download the train, validation and tests datasets and place them in the same folder as the training resp. evaluation script.

For the evaluation script the **test.csv** file in **predictions** is also needed. It is used for grouping documents into folders. Place it in the same folder as the script.

Supported command line arguments:
- Use the **--clear** flag to reset the created cache file in both the training and evaluation script.
- Use the **--resume** flag to continue training from the last training run (after last completed epoch).
- Use **--batch** to specify the batch size to be used in training.

Run the scripts in the docker container or install the dependencies yourself.
The following commands work for Linux systems and were tested with Python 3.9:
    
    pip install -r requirements.txt
    apt-get -y install tesseract-ocr
    apt-get install poppler-utils -y

## Final Model
You can find the final model on HuggingFace: https://huggingface.co/qrauli/pss-business-documents