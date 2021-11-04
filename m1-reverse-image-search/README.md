
# Usage
We are currently using reverse image search on [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html).

    pip install -r requirements.txt
    
Create `.env` file and add variables that point to directory of CIFAR100 pickle file & CIFAR100 images directory as follows:

    DATA_DIR=<value>
    IMGS_DIR=<value>

Pass image path to `main.py` as an argument. For example,
   

    python main.py path/to/image.jpg
