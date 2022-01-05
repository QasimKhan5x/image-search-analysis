# Image Search Analysis

Content-based image analysis, retrieval, and search using visual features, metadata, and natural language.

Currently, we have developed a command line tool and web app that allows you to search for images similar to a query image in your favorite dataset. To proceed further, first install milvus by following the instructions below. Then proceed to the `m1_reverse_image_search` folder for further instructions.

To allow metadata creation of an image, we have provided the `Metadata_Creation.ipynb` notebook. Open that notebook in Google Colab with GPU on. Then, use `create_metadata_from_imgpath` to create metadata from the path to an image.

## Using milvus

To setup milvus, you need docker.

### Start Milvus

`docker-compose up -d`

### Stop Milvus

`docker-compose down`
