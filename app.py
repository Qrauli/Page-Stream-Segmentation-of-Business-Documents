from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import zipfile
from PyPDF2 import PdfReader, PdfWriter
from contextlib import asynccontextmanager
from tensorflow import keras

tags_metadata = [
    {
        "name": "prediction",
        "description": "Attaching a PDF File will split it into individual documents and return them in a zip file",
    }
]


model = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model
    model = keras.models.load_model('path/to/location')
    yield
    # Clean up the ML models and release the resources
    keras.backend.clear_session()

app = FastAPI(title="API", description="API for PSS", version="1.0", openapi_tags=tags_metadata, lifespan=lifespan)

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
                start = i+1

            # Add the current page to the PdfFileWriter object
            pdf_writer.add_page(pdf_reader.pages[i])

            if i == len(splits) - 1 or splits[i+1] == 1:
                end = i+1
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
    # temporary split calculation
    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    count = len(pdf_reader.pages)
    categories = []
    for i in range(count):
        categories.append(1)
    return categories
