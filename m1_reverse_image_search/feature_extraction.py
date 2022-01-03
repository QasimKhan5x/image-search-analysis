import torch
from torchvision.models import efficientnet_b7
from torchvision.models.feature_extraction import create_feature_extractor


class FExt(torch.nn.Module):
    def __init__(self):
        super(FExt, self).__init__()
        # create feature extractor
        model = efficientnet_b7(pretrained=True).eval()
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 4 feature vectors
        return_nodes = {
            # node_name: user-specified key for output dict
            # 'features.2.6.block.2.avgpool': 'low',
            # 'features.5.4.block.2.avgpool': 'middle',
            # 'features.7.3.block.2.avgpool': 'high',
            'flatten': 'final',
        }
        self.fx = create_feature_extractor(model, return_nodes=return_nodes)
        self.fx = self.fx.to(self.device)

    def forward(self, x):
        result = self.fx(x)
        for layer in result:
            if layer == 'final':
                break
            else:
                # BxCxWxH --> BxC
                result[layer] = result[layer].squeeze(-1).squeeze(-1)
        return result


if __name__ == '__main__':
    model = FExt()
    x = torch.randn(1, 3, 224, 224).cuda()
    y = model(x)
    for layer in y:
        y[layer] = y[layer].cpu().detach().numpy()
        print(y[layer].shape)
