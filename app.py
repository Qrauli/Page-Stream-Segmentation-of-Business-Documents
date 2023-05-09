from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import zipfile
from PyPDF2 import PdfReader, PdfWriter
from contextlib import asynccontextmanager
from tensorflow import keras
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from models.training import image_to_tensor, tokenize_from_ocr
import tensorflow as tf
import numpy as np
from transformers import TFLayoutLMModel, LayoutLMTokenizer
from huggingface_hub import from_pretrained_keras
from config import settings

tags_metadata = [
    {
        "name": "prediction",
        "description": "Attaching a PDF File will split it into individual documents and return them in a zip file",
    }
]

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model
    global tokenizer
    model = from_pretrained_keras("qrauli/pss-business-documents", custom_objects={"TFLayoutLMModel": TFLayoutLMModel})
    # model = "t"
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    yield
    # Clean up the ML models and release the resources
    keras.backend.clear_session()
    # model = "r"


app = FastAPI(title="API", description="API for PSS", version="1.0", openapi_tags=tags_metadata, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/predict', tags=["prediction"])
async def get_prediction(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(400, detail="Invalid document type")

    pdf_content = await file.read()
    splits = await predict(pdf_content)
    zip_stream = io.BytesIO()

    with zipfile.ZipFile(zip_stream, mode="w") as zip_file:
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        pdf_writer = PdfWriter()
        start = 1
        end = len(splits)
        for i in range(len(splits)):
            if splits[i] == 1:
                pdf_writer = PdfWriter()
                start = i + 1

            # Add the current page to the PdfFileWriter object
            pdf_writer.add_page(pdf_reader.pages[i])

            if i == len(splits) - 1 or splits[i + 1] == 1:
                end = i + 1
                # Create an in-memory byte stream to hold the new PDF file contents
                pdf_stream = io.BytesIO()

                # Write the new PDF file to the in-memory byte stream
                pdf_writer.write(pdf_stream)

                # Set the position of the in-memory byte stream to the beginning
                pdf_stream.seek(0)

                # Add the new PDF file to the zip file
                zip_file.writestr(f"page_{start}-{end}.pdf", pdf_stream.getvalue())

    zip_stream.seek(0)

    # Create a StreamingResponse object with the in-memory byte stream as its content
    response = StreamingResponse(zip_stream, media_type="application/zip")

    # Set the Content-Disposition header to indicate the filename of the zip file
    response.headers["Content-Disposition"] = f"attachment; filename={file.filename}.zip"

    return response


async def predict(pdf_content: bytes):
    """calculates an array of split prediction values for every page of the given pdf"""
    global tokenizer
    global model
    images = convert_from_bytes(pdf_content, grayscale=True)

    # convert each image to input data
    input_data = []

    for image in images:
        pil_image = image
        image_tensor = image_to_tensor(tf.convert_to_tensor(pil_image), (224, 224))
        pil_image = pil_image.filter(ImageFilter.MedianFilter())
        pil_image = pil_image.resize((1000, 1000))
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(2)
        pil_image = pil_image.convert('1')
        dataframe = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DATAFRAME)
        input_ids, bbox, attention_mask, token_type_ids = tokenize_from_ocr("", tokenizer, 100, dataframe)
        input_data.append(
            {"image": image_tensor, "input_ids": input_ids, "bbox": bbox, "attention_mask": attention_mask,
             "token_type_ids": token_type_ids})

    batch_generator = tf.data.Dataset.from_generator(lambda: generator(input_data),
                                                     output_types=(tf.int32, tf.int32, tf.int32, tf.int32, tf.int64))
    iterator = iter(batch_generator.batch(settings.batch_size).take(-1))
    last_elem = None
    classification = None
    while True:
        try:
            input_ids, bbox, attention_mask, token_type_ids, image = next(iterator)
            if last_elem:
                input_ids_l, bbox_l, attention_mask_l, token_type_ids_l, image_l = last_elem
                input_ids = tf.concat([tf.expand_dims(input_ids_l, 0), input_ids], axis=0)
                bbox = tf.concat([tf.expand_dims(bbox_l, 0), bbox], axis=0)
                attention_mask = tf.concat([tf.expand_dims(attention_mask_l, 0), attention_mask], axis=0)
                token_type_ids = tf.concat([tf.expand_dims(token_type_ids_l, 0), token_type_ids], axis=0)
                image = tf.concat([tf.expand_dims(image_l, 0), image], axis=0)
            logits = model([input_ids, bbox, attention_mask, token_type_ids, image])
            if last_elem:
                logits = logits[1:]
            last_elem = (input_ids[-1], bbox[-1], attention_mask[-1], token_type_ids[-1], image[-1])
            logits = logits.numpy()
            if logits.ndim == 1:
                logits = np.expand_dims(logits, axis=0)
            if classification is not None:
                classification = np.append(classification, logits, 0)
            else:
                classification = logits
        except tf.errors.OutOfRangeError:
            break
        except StopIteration as e:
            break
        except ValueError as e:
            print(e)
            continue
        except tf.errors.ResourceExhaustedError as e:
            print("Not enough memory for running model prediction: Resources exhausted")
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            count = len(pdf_reader.pages)
            categories = []
            for i in range(count):
                categories.append(1)
            return categories

    categories = []
    for classi in classification:
        if classi[0] > classi[1]:
            categories.append(1)
        else:
            categories.append(0)
    categories[0] = 1
    return categories


def generator(input_data):
    """generator returning extracted data of single elements"""
    i = 0
    while i < len(input_data):
        input_ids = tf.convert_to_tensor(np.array(input_data[i]["input_ids"]))
        bbox = tf.convert_to_tensor(np.array(input_data[i]["bbox"]))
        attention_mask = tf.convert_to_tensor(np.array(input_data[i]["attention_mask"]))
        token_type_ids = tf.convert_to_tensor(np.array(input_data[i]["token_type_ids"]))
        image = tf.convert_to_tensor(np.array(input_data[i]["image"]))

        yield input_ids, bbox, attention_mask, token_type_ids, image
        i += 1
