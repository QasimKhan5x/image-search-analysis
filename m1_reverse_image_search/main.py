import argparse
import os
import sqlite3

from dotenv import load_dotenv
from PIL import Image
from pymilvus import connections

from feature_collection import get_collection
from inference import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="the path to your query image")
    args = parser.parse_args()
    load_dotenv()

    imgs_dir = os.getenv("DATA_DIR")
    query_img_path = args.img_path
    img = Image.open(query_img_path).convert("RGB")

    # connect to Milvus
    connections.connect(host="127.0.0.1", port=19530)
    con = sqlite3.connect('image_paths.db')
    cur = con.cursor()

    print('Loading Collection...')
    start = time.time()
    collection = get_collection()
    # Load the collection to memory before conducting a vector similarity search
    collection.load()
    end = time.time() - start
    print(f'Loading Collection took {end} seconds')

    print('Performing model inference')
    start = time.time()
    embedding = get_embedding(img, 'final')
    end = time.time() - start
    print(f'Inference took {end} seconds')
    filepaths = get_nn_filepaths(cur, embedding, collection)

    # milvus
    connections.disconnect("default")
    # sqlite
    con.close()
