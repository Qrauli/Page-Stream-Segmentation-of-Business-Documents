from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import zipfile

app = FastAPI(title="API", description="API for PSS", version="1.0")

@app.post('/predict', tags=["predictions"])
async def get_prediction(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(400, detail="Invalid document type")

    pdf_content = await file.read()
    zip_stream = io.BytesIO()

    with zipfile.ZipFile(zip_stream, mode="w") as zip_file:
        # Add the PDF file to the zip file
        zip_file.writestr(file.filename, pdf_content)

    zip_stream.seek(0)

    # Create a StreamingResponse object with the in-memory byte stream as its content
    response = StreamingResponse(zip_stream, media_type="application/zip")

    # Set the Content-Disposition header to indicate the filename of the zip file
    response.headers["Content-Disposition"] = f"attachment; filename={file.filename}.zip"

    return response