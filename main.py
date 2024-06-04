import dill
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
with open('model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int


class Prediction(BaseModel):
    id: int
    price: int
    Result: str


@app.get('/status')
def status():
    return "I'm alive!"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    try:
        form.posting_date = str(form.posting_date)
        df = pd.DataFrame([form.dict()])
        y = model['model'].predict(df)

        return {
            'id': form.id,
            'price': form.price,
            'Result': y[0]
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc}")
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(status_code=500, content={"message": "Something went wrong"})
