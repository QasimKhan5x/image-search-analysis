import torch
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor


class FExt(torch.nn.Module):
    def __init__(self):
        super(FExt, self).__init__()
        # create feature extractor
        model = resnet18(pretrained=True).eval()
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(self.device)
        # extract flatten layer and rename it as features_512
        return_nodes = {"flatten": "features_512"}
        self.fx = create_feature_extractor(model, return_nodes=return_nodes)

    def forward(self, x):
        y = self.fx(x)
        return y['features_512']
