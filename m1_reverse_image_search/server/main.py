import sqlite3
import sys  # noqa

sys.path.append('../')  # noqa

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from inference import get_nn_filepaths
from PIL import Image
from pymilvus import connections

app = FastAPI()
# connect to Milvus
connections.connect(host="127.0.0.1", port=19530)
con = sqlite3.connect('../image_paths.db')
cur = con.cursor()


@app.post("/uploadFile")
async def create_upload_files(resultLimit: int = Form(...), file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        res = get_nn_filepaths(cursor=cur, img=img, topK=resultLimit)
    except:
        res = None
    finally:
        return res


app.mount('/', StaticFiles(directory='static', html=True), name='static')