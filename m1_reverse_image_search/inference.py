import time

import torch

from feature_collection import search_collection
from feature_extraction import FExt
from util import transform_PIL

model = FExt()


def get_embedding(img, transform, field='final'):
    '''Create vector embeddings based on layer specified in `field`'''
    assert field in ('low', 'middle', 'high', 'final')

    with torch.no_grad():
        if transform:
            tensor = transform_PIL(img)
            tensor = torch.unsqueeze(tensor, 0)
        else:
            tensor = img
        tensor = tensor.cuda() if torch.cuda.is_available() else tensor
        embeddings = model(tensor)
        for layer in embeddings:
            embeddings[layer] = embeddings[layer].cpu(
            ).detach().numpy().tolist()
        required_embedding = embeddings[field]
    return required_embedding


def get_nn_filepaths(cursor, embeddings, collection,
                     topK=10, field='final'):
    '''
    Gets the filepaths of the topK images 
    most similar to a query image.

    Parameters:
            cursor (sqlite3.Cursor): Cursor that points to local sqlite3 database
            embeddings (torch.Tensor): Embedding of query image
            collection (milvus collection): Collection that contains vector embeddings
            topK (int): Number of neighbors to return (default 10)
            field (str): Which vector embedding to use
    Returns:
            filepaths (list): List containing string filepath strings that
                              are the nearest neighbors of the query image
    '''
    # get 1st index element because only 1 image was passed as query
    results = search_collection(collection, embeddings, topK, field)[0]
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
