from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from PIL import Image
import pickle
root_path = "/home/quan/code/qnguyen-gh/smArt/smArt"
#root_path = requests.get("https://storage.googleapis.com/artdataset",stream=True)
import sys; sys.path
sys.path.append(root_path)
from smArt.trainer import Trainer
from google.cloud import storage

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

    X_pred = Image.open(requests.get(f'https://storage.googleapis.com/artdataset/wikiart_sample/{genre}/{filename}', stream=True).raw)
    ## what I tried
    data_file = 'test.sav'
    client = storage.Client().bucket("artdataset")
    blob = client.blob(data_file)
    blob.download_to_filename(data_file)
    loaded_model = pickle.load(open(data_file, "rb"))
    # loaded_model = pickle.load(open("/home/quan/code/qnguyen-gh/smArt/test.sav", 'rb'))
    size = 128, 128
    y_pred = loaded_model.predict_image(X_pred,size)
    return y_pred


# @app.post("/uploadfile")
# async def create_upload_file(file: UploadFile=File(...)):
#     #url = f"http://127.0.0.1:8000/predict{file=}"
#     #return requests.get(url).json()[0]
#     return {"file": file.filename}
