from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import joblib
import requests
import os
import uuid
from PIL import Image
import pickle
root_path = "/Users/olganowak/code/olganowak/smArt/smArt/"
import sys; sys.path
sys.path.append(root_path)
from smArt.trainer import Trainer

app = FastAPI()

img=[]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello there"}

# @app.get("/upload/")
# async def create_upload_file(file: UploadFile=File(...)):
#     file.filename = f"{uuid.uuid4()}.jpg"
#     content = await file.read()
#     img.append(content)
#     return {"filename": file.filename}

# @app.get("/predict/")
# def predict():
#     response = Response(content=img[0])
#     X_pred = response
#     model = joblib.load('model.joblib')
#     y_pred=model.predict(X_pred)
#     return y_pred

#from tkinter import Tk     # from tkinter import Tk for Python 3.x
#from tkinter.filedialog import askopenfilename

@app.get("/predict/")
def predict(genre, filename):
    # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # filename = askopenfilename()
    y_pred=1
    X_pred = Image.open(requests.get(f'https://storage.googleapis.com/artdataset/wikiart_sample/{genre}/{filename}', stream=True).raw)
    #X_pred = f'https://storage.googleapis.com/artdataset/wikiart_sample/{genre}/{filename}'
    loaded_model = pickle.load(open("/Users/olganowak/code/olganowak/smArt/notebooks/15_genres.sav", 'rb'))
    #model = joblib.load('/Users/olganowak/code/olganowak/smArt/notebooks/trainer.joblib')
    size = 128, 128
    y_pred = loaded_model.predict_image(X_pred,size)
    return y_pred

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# @app.get("/")
# def index():
#     return {"greeting": "Hello there"}

# @app.get("/predict")
# def predict(some_file_path):
#     X_pred = some_file_path
#     model = joblib.load('model.joblib')
#     y_pred=model.predict(X_pred)
#     return y_pred

# @app.post("/uploadfile")
# async def create_upload_file(file: UploadFile=File(...)):
#     #url = f"http://127.0.0.1:8000/predict{file=}"
#     #return requests.get(url).json()[0]
#     return {"file": file.filename}
