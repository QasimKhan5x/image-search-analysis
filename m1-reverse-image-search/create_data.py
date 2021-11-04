import os

import torch
import torchvision.transforms as T
from dotenv import load_dotenv
from pymilvus import connections
from torchvision.datasets import CIFAR100

from feature_collection import get_collection
from feature_extraction import FExt

if __name__ == '__main__':
    # connect to Milvus
    connections.connect(host="127.0.0.1", port=19530)
    feature_extractor = FExt()
    load_dotenv()
    tensors_dir = os.getenv('DATA_DIR')
    imgs_dir = os.getenv('IMGS_DIR')

    tensor_ds = CIFAR100(tensors_dir,
                         transform=T.Compose([
                             T.Resize(256),
                             T.CenterCrop(224),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                         ])
                         )
    dataloader = torch.utils.data.DataLoader(
        tensor_ds, num_workers=4, batch_size=256, shuffle=False, pin_memory=True)
    collection = get_collection()

    for i, (inputs, labels) in enumerate(dataloader):
        with torch.no_grad():
            output = feature_extractor.get_features(
                inputs, transform=False).squeeze()
            output = output.cpu().detach().numpy()

        # TODO
        # insert into Db
        mr = collection.insert([output.tolist()])
        ids = mr.primary_keys
