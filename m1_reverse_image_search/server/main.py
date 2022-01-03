import json
import sqlite3
import sys  # noqa
import time

sys.path.append('../')  # noqa

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from feature_collection import get_collection
from inference import get_embedding, get_nn_filepaths
from PIL import Image
from pymilvus import connections

app = FastAPI()
# connect to Milvus
connections.connect(host="127.0.0.1", port=19530)
con = sqlite3.connect('../image_paths.db')
cur = con.cursor()
collection = get_collection()
# Load the collection to memory before conducting a vector similarity search
print('Loading Collection...')
start = time.time()
collection.load()
end = time.time() - start
print(f'Loading Collection took {end} seconds')


@app.get('/closeServer')
def closeServer():
    # Release the collection loaded in Milvus to reduce
    # memory consumption when the search is completed.
    collection.release()
    # sqlite3 connection
    # con.close()
    # milvus connection
    connections.disconnect("default")
    return 'Server shutdown successfully'

embeddings = None

@app.post("/uploadFile")
async def create_upload_files(resultLimit: int = Form(...), old: bool = Form(...), img_file: UploadFile = File(...), json_file: UploadFile = File(...)):
    try:
        img = Image.open(img_file.file).convert("RGB")
        if not old: 
            embeddings = get_embedding(img, True)
        res = get_nn_filepaths(cursor=cur,
                               embeddings=embeddings,
                               collection=collection,
                               topK=resultLimit)
        x = json_file.file.read().decode('utf-8')
        print(json.loads(x))
    except Exception as e:
        print(e)
        res = None
    finally:
        return res

app.mount('/', StaticFiles(directory='static', html=True), name='static')
