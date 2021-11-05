# Usage

We are currently using reverse image search on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

First, install necessary requirements via

    pip install -r requirements.txt

Create `.env` file and add a variable that points to the directory of PASCAL VOC 2012 images directory:

    DATA_DIR=<value>

Then follow the subsequent steps by entering the non-commented (non-#) lines in your terminal:

    # create milvus collection
    python feature_collection.py
    # store embeddings in collection and
    # create and populate sqlite3 database
    python create_data.py

Pass query image path to `main.py` as an argument. For example,

    python main.py path/to/image.jpg
