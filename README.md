# Image Search Analysis

Content-based image analysis, retrieval, and search using keywords as well as visual and textual features.

Currently, we have developed a command line tool and web app that allows you to search for images similar to a query image in your favorite dataset. To proceed further, first install milvus by following the instructions below. Then proceed to the `m1_reverse_image_search` folder for further instructions.

To allow metadata creation of an image, we have provided the `Metadata_Creation.ipynb` notebook. Open that notebook in Google Colab with GPU on. Then, use `create_metadata_from_imgpath` to create metadata from the path to an image.

## Using milvus

To setup milvus, you need [docker](https://docs.docker.com/get-docker/).

### Start Milvus

`docker-compose up -d`

This will generate a `volumes` folder in the root directory.

To check if Milvus is running, enter 

`docker-compose ps` 

3 containers should appear running healthy. More information [here](https://milvus.io/docs/install_standalone-docker.md).

### Stop Milvus

`docker-compose down`

An issue we are facing with Milvus currently is that if you stop Milvus, subsequent searches 
after restarting Milvus will take several minutes. Therefore, restarting Milvus means
following the entire process of creating a Milvus collection from scratch after deleting
the `volumes` folder.
