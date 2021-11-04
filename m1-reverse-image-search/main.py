import argparse

from pymilvus import connections

from inference import get_nn

parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="the path to your query image")
args = parser.parse_args()
query_img_path = args.img_path

# connect to Milvus
connections.connect(host="127.0.0.1", port=19530)
result = get_nn(query_img_path)
