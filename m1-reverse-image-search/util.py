import torchvision.transforms as T


def transform_PIL(img):
    '''
    img -> PIL Image
    '''
    transform_ops = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])]
    )
    return transform_ops(img)
