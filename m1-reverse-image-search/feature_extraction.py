import torch
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor

from util import transform_PIL


class FExt:
    def __init__(self):
        # create feature extractor
        model = resnet18().eval()
        return_nodes = {"flatten": "features_512"}
        self.fx = create_feature_extractor(model, return_nodes=return_nodes)

    def get_features(self, x, transform=True):
        # convert PIL img to tensor
        if transform:
            x = transform_PIL(x)
            x = torch.unsqueeze(x, 0)
        y = self.fx(x)
        return y['features_512']
