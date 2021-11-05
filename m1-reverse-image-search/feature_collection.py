import logging
import time

import pymilvus
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections)


def setup_collection(collection_name="voc2012_ris"):
    dim = 512
    default_fields = [
        FieldSchema(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    default_schema = CollectionSchema(
        fields=default_fields, description="PASCAL VOC 2012 collection")

    collection = Collection(name=collection_name, schema=default_schema)
    collection.load()
    return collection


def get_collection(collection_name="voc2012_ris"):
    '''Assumes that a connection to milvus has been established'''
    assert pymilvus.utility.get_connection().has_collection(collection_name), \
        "ERROR: Collection not found"

    return setup_collection(collection_name)


def search_collection(collection, vectors, topK=50):
    search_params = {"metric_type": "L2", "params": {"nprobe": 32}}
    start = time.time()
    if not isinstance(vectors, list):
        vectors = vectors.tolist()
    res = collection.search(vectors, "vector",
                            param=search_params, limit=topK, expr=None)
    end = time.time() - start
    logging.info(f"Search took {end} seconds")
    return res


if __name__ == '__main__':
    # connect to Milvus
    connections.connect(host="127.0.0.1", port=19530)
    # Create a collection
    collection = setup_collection()
    # Create IVF_SQ8 index to the  collection
    default_index = {"index_type": "IVF_SQ8",
                     "params": {"nlist": 2048}, "metric_type": "L2"}
    collection.create_index(field_name="vector", index_params=default_index)
    collection.load()
    print("SUCCESS")
