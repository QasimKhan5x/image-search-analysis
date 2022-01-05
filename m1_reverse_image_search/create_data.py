import os
import sqlite3

from dotenv import load_dotenv
from PIL import Image
from pymilvus import connections
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from feature_collection import get_collection
from inference import get_embedding
from util import transform_PIL


class VOC2012(Dataset):
    def __init__(self):
        load_dotenv()
        self.img_dir = os.getenv('DATA_DIR')
        self.image_names = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index, transform=True):
        img_path = os.path.join(self.img_dir, self.image_names[index])
        img = Image.open(img_path).convert("RGB")
        if transform:
            tensor = transform_PIL(img)
            return tensor, self.image_names[index]
        return img, self.image_names[index]


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
    dataset = VOC2012()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    collection = get_collection()
    for img, filenames in tqdm(dataloader):
        embeddings = get_embedding(img, transform=False)
        # insert into milvus
        mr = collection.insert([embeddings])
        # get milvus id
        mr_ids = mr.primary_keys
        records = list(zip(mr_ids, filenames))
        # insert into sqlite
        cur.executemany('INSERT INTO paths VALUES(?,?);', records)

    print('Inserted Data. Creating Indexes...')
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }

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
