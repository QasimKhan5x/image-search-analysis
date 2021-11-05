import argparse
import os
import sqlite3

from dotenv import load_dotenv
from pymilvus import connections

from inference import get_nn

parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="the path to your query image")
args = parser.parse_args()
load_dotenv()
imgs_dir = os.getenv("DATA_DIR")
query_img_path = args.img_path

# connect to Milvus
connections.connect(host="127.0.0.1", port=19530)
con = sqlite3.connect('image_paths.db')
cur = con.cursor()
results = get_nn(query_img_path)[0]

for result in results:
    cur.execute(f'SELECT path FROM paths WHERE id = {result.id}')
    rows = cur.fetchall()
    filename = rows[0][0]
    filepath = os.path.join(imgs_dir, filename)
    print(filepath, result.distance)
