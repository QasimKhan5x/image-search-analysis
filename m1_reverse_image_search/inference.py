import logging
import time

import torch
from PIL import Image

from feature_collection import get_collection, search_collection
from feature_extraction import FExt
from util import transform_PIL

model = FExt()


def get_nn(tensor, topK=10, field='final'):
    '''
    Input:
        tensor: a tensor representing an image to query
        field (str): Which vector embedding to use
    Output:
        results: array with elements having attribrutes (id, distance)
    '''
    assert field in ('low', 'middle', 'high', 'final')
    collection = get_collection()
    # Load the collection to memory before conducting a vector similarity search
    print('Loading Collection...')
    start = time.time()
    collection.load()
    end = time.time() - start
    print(f'Loading Collection took {end} seconds')
    # Create vector embeddings
    print('Performing model inference')
    start = time.time()
    embeddings = model(tensor)
    for layer in embeddings:
        embeddings[layer] = embeddings[layer].cpu().detach().numpy().tolist()
    required_embedding = embeddings[field]
    end = time.time() - start
    print(f'Inference took {end} seconds')
    # get 1st index element because only 1 image was passed as query
    results = search_collection(collection, required_embedding, topK, field)[0]
    # Release the collection loaded in Milvus to reduce memory consumption when the search is completed.
    collection.release()
    return results


def get_nn_filepaths(cursor, img=None, img_path=None, topK=10, field='final'):
    '''
    Gets the filepaths of the topK images 
    most similar to a query image.

    Parameters:
            cursor (sqlite3.Cursor): Cursor that points to local sqlite3 database
            img_path (string): Path to query image
            img (PIL.Image.Image): PIL Image object used as query image 
            topK (int): Number of neighbors to return (default 10)
            field (str): Which vector embedding to use
    Returns:
            filepaths (list): List containing string filepath strings that
                              are the nearest neighbors of the query image
    '''
    if img == None and img_path == None:
        logging.error("Pass either PIL.Image.Image OR image file path")
    if img_path:
        img = Image.open(img_path).convert("RGB")
    tensor = transform_PIL(img)
    tensor = torch.unsqueeze(tensor, 0)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    tensor = tensor.to(device)
    results = get_nn(tensor, topK, field)
    filepaths = []
    for result in results:
        cursor.execute(f'SELECT path FROM paths WHERE id = {result.id}')
        rows = cursor.fetchall()
        # Only 1 neighbor per result
        filename = rows[0][0]
        # Filename distance
        print(f"{filename} {result.distance}")
        filepaths.append(filename)
    return filepaths
