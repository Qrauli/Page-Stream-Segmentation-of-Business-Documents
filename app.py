from fastapi import FastAPI

app = FastAPI(title="API", description="API for PSS", version="1.0")

@app.get('/predict', tags=["predictions"])
async def get_prediction():
    return {"prediction": "test"}