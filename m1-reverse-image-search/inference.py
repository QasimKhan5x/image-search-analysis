import torch
from PIL import Image

from feature_collection import *
from feature_extraction import FExt

model = FExt()


def get_nn(img_path, topK=10):
    '''
    Input:
        img_path: path to query image
    Output:
        results: array with elements having attribrutes (id, distance)
    '''
    collection = get_collection()
    img = Image.open(img_path)
    embeddings = model.get_features(img).detach().numpy().tolist()
    results = search_collection(collection, embeddings, topK)
    return results
