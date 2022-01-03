import time

from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)


def setup_collection(collection_name):
    low = 288
    mid = 1344
    high = 3840
    final = 2560

    img_id = FieldSchema(
        name="img_id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )

    # low_level = FieldSchema(
    #     name="low_level_features",
    #     dtype=DataType.FLOAT_VECTOR,
    #     dim=low
    # )

    # mid_level = FieldSchema(
    #     name="mid_level_features",
    #     dtype=DataType.FLOAT_VECTOR,
    #     dim=mid
    # )

    # high_level = FieldSchema(
    #     name="high_level_features",
    #     dtype=DataType.FLOAT_VECTOR,
    #     dim=high
    # )

    final_level = FieldSchema(
        name="final_layer_features",
        dtype=DataType.FLOAT_VECTOR,
        dim=final
    )

    # schema = CollectionSchema(
    #     fields=[img_id, low_level, mid_level, high_level, final_level],
    #     description="VOC2012 EfficientNet-B07"
    # )

    schema = CollectionSchema(
        fields=[img_id, final_level],
        description="VOC2012 EfficientNet-B07"
    )

    collection = Collection(name=collection_name, schema=schema,
                            using='default', shards_num=4)
    return collection


def get_collection(collection_name="voc2012_effnetb07"):
    '''Assumes that a connection to milvus has been established'''
    assert utility.has_collection(collection_name), \
        "ERROR: Collection not found"
    collection = Collection(collection_name)
    return collection


def search_collection(collection, vectors, topK, field='final'):
    assert field in ('low', 'middle', 'high', 'final')
    if field == 'low':
        field = 'low_level_features'
    elif field == 'middle':
        field = 'mid_level_features'
    elif field == 'high':
        field = 'high_level_features'
    else:
        field = 'final_layer_features'

    # search_params = {"metric_type": "L2", "params": {"nq": 10}}
    search_params = {"metric_type": "L2"}
    if not isinstance(vectors, list):
        vectors = vectors.tolist()
    print("Searching collection...")
    start = time.time()
    res = collection.search(data=vectors,
                            anns_field=field,
                            param=search_params,
                            limit=topK)
    end = time.time() - start
    print(f"Search took {end} seconds")
    return res


if __name__ == '__main__':
    # connect to Milvus
    connections.connect(host="127.0.0.1", port=19530)
    # Create or Get a collection
    if utility.has_collection("voc2012_effnetb07"):
        print("Collection Found!")
        collection = get_collection()

    else:
        print("Collection not found. Creating.")
        collection = setup_collection("voc2012_effnetb07")
    connections.disconnect("default")
    print("SUCCESS")
