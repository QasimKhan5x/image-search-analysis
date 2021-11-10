import logging
import os

from PIL import Image

from feature_collection import get_collection, search_collection
from feature_extraction import FExt

model = FExt()


def get_nn(img, topK=10):
    '''
    Input:
        img_path: path to query image
    Output:
        results: array with elements having attribrutes (id, distance)
    '''
    collection = get_collection()
    if isinstance(img, str):
        img = Image.open(img)
    else:
        assert isinstance(img, Image.Image)
    embeddings = model.get_features(img).detach().numpy().tolist()
    # get 1st index element because only 1 image was passed as query
    results = search_collection(collection, embeddings, topK)[0]
    return results


def get_nn_filepaths(imgs_dir, cursor, img=None, img_path=None, topK=10):
    '''
    Gets the filepaths of the topK images 
    most similar to a query image.

    Parameters:
            imgs_dir (string): The path to the directory of images
            cursor (sqlite3.Cursor): Cursor that points to local sqlite3 database
            topK (int): Number of neighbors to return (default 10)
            img_path (string): Path to query image
            img (PIL.Image.Image): PIL Image object used as query image 
    Returns:
            filepaths (list): List containing string filepath strings in the imgs_dir
                              that are the nearest neighbors of the query image
    '''
    if img == None and img_path == None:
        logging.error("Pass either PIL.Image.Image OR image file path")
    if img_path:
        img = Image.open(img_path)
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
