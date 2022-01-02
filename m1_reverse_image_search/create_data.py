import os
import sqlite3

import torch
from dotenv import load_dotenv
from PIL import Image
from pymilvus import connections
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from feature_collection import get_collection
from feature_extraction import FExt
from util import transform_PIL


class VOC2012(Dataset):
    def __init__(self):
        load_dotenv()
        self.img_dir = os.getenv('DATA_DIR')
        self.image_names = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_names[index])
        img = Image.open(img_path)
        tensor = transform_PIL(img)
        return tensor, self.image_names[index]


def create_table():
    con = sqlite3.connect('image_paths.db')
    cur = con.cursor()
    # Create table
    cur.execute('''CREATE TABLE paths
                (id unsigned big int, path text)''')

    # Save (commit) the changes
    con.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    con.close()


def populate_db():
    con = sqlite3.connect('image_paths.db')
    cur = con.cursor()

    feature_extractor = FExt()
    dataset = VOC2012()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    collection = get_collection()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    for tensor, filenames in tqdm(dataloader):
        with torch.no_grad():
            tensor = tensor.to(device)
            result = feature_extractor(tensor)
            for layer in result:
                result[layer] = result[layer].cpu().detach().numpy().tolist()

        # insert into Db
        mr = collection.insert([
            result['low'],
            result['middle'],
            result['high'],
            result['final']
        ])
        mr_ids = mr.primary_keys
        records = list(zip(mr_ids, filenames))
        cur.executemany('INSERT INTO paths VALUES(?,?);', records)
    print('Inserted Data. Creating Indexes...')
    index_params = {
        "metric_type": "L2",
        "index_type": "FLAT",
        "params": {"nlist": 523}
    }
    collection.create_index(
        field_name="low_level_features",
        index_params=index_params
    )
    collection.create_index(
        field_name="mid_level_features",
        index_params=index_params
    )
    collection.create_index(
        field_name="high_level_features",
        index_params=index_params
    )
    collection.create_index(
        field_name="final_layer_features",
        index_params=index_params
    )
    print('Indexes Created!')

    # commit the changes to db
    con.commit()
    print('Data added to DB')
    # close the connection
    con.close()


if __name__ == '__main__':
    # connect to Milvus
    connections.connect(host="127.0.0.1", port=19530)
    if not os.path.isfile("image_paths.db"):
        create_table()  # raises error if db already exists
        populate_db()  # takes around 5 min to run
    connections.disconnect("default")
