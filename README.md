# Image Search Analysis

## Overview

Content-based image analysis, retrieval, and search, utilizing both keywords and a combination of visual and textual features. By analyzing image content, the system can efficiently retrieve and categorize images based on specified keywords or inherent visual properties.
 
Currently, we have developed a command line tool and web app that allows you to search for images similar to a query image in your favorite dataset. To proceed further, first install milvus by following the instructions below. Then proceed to the `m1_reverse_image_search` folder for further instructions.

To allow metadata creation of an image, we have provided the `Metadata_Creation.ipynb` notebook. Open that notebook in Google Colab with GPU on. Then, use `create_metadata_from_imgpath` to create metadata from the path to an image.

## Getting Started

### Using milvus

To setup milvus, you need [docker](https://docs.docker.com/get-docker/).

#### Start Milvus

`docker-compose up -d`

This will generate a `volumes` folder in the root directory.

To check if Milvus is running, enter 

`docker-compose ps` 

3 containers should appear running healthy. More information [here](https://milvus.io/docs/install_standalone-docker.md).

#### Stop Milvus

`docker-compose down`

An issue we are facing with Milvus currently is that if you stop Milvus, subsequent searches 
after restarting Milvus will take several minutes. Therefore, after stopping Milvis you should,

1. Delete the `volumes` folder and the `image_paths.db` file if you have created it
2. Start Milvus
3. Create a milvus collection and the sqlite3 database again
