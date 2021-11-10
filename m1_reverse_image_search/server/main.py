import sqlite3
import sys

sys.path.append('../')

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from inference import get_nn_filepaths
from PIL import Image
from pymilvus import connections
from starlette.requests import Request

app = FastAPI()
# connect to Milvus
connections.connect(host="127.0.0.1", port=19530)
con = sqlite3.connect('image_paths.db')
cur = con.cursor()

templates = Jinja2Templates(directory='templates')


@app.post("/uploadfiles/", response_class=HTMLResponse)
async def create_upload_files(request: Request, resultLimit: int = Form(...), file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    res = get_nn_filepaths(imgs_dir='./static/images/', cursor=cur, img=img, topK=resultLimit)
    return templates.TemplateResponse('results.html',
                                      {"request": request,
                                       "list": res,
                                       "limit": resultLimit}
                                      )


app.mount('/', StaticFiles(directory='static', html=True), name='static')
