import os

from PIL import Image

from feature_collection import get_collection, search_collection
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
    # get 1st index element because only 1 image was passed as query
    results = search_collection(collection, embeddings, topK)[0]
    return results


def get_nn_filepaths(img_path, imgs_dir, cursor, topK=10):
    '''
    Gets the filepaths of the topK images 
    most similar to a query image.

    Parameters:
            imgs_dir (string): The path to the directory of images
            img_path (string): Path to query image
            cursor (sqlite3.Cursor): Cursor that points to local sqlite3 database
            topK (int): Number of neighbors to return (default 10)
    Returns:
            filepaths (list): List containing string filepath strings in the imgs_dir
                              that are the nearest neighbors of the query image
    '''
    results = get_nn(img_path, topK)
    filepaths = []
    for result in results:
        cursor.execute(f'SELECT path FROM paths WHERE id = {result.id}')
        rows = cursor.fetchall()
        # Only 1 neighbor per result
        filename = rows[0][0]
        filepath = os.path.join(imgs_dir, filename)
        filepaths.append(filepath)
    return filepaths
