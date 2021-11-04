import torch
from feature_collection import *
from feature_extraction import FExt
from PIL import Image
from util import transform_PIL

model = FExt()

def get_nn(img_path):
    '''
    Input:
        img_path: path to query image
    Output:
        results: array with elements having attribrutes (id, distance)
    '''
    collection = get_collection()
    img = Image.open(img_path)
    embeddings = torch.flatten(model.get_features(img)).detach().numpy()
    results = search_collection(collection, embeddings)
    return results
