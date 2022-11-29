

ViT_parameters = {
    "image_size": 256,
    "patch_size": 32,
    "num_classes": 1000,
    "dim": 1024,
    "depth": 6,
    "heads": 16,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1
}

T2TViT_parameters = {
    "dim": 512,
    "image_size": 224,
    "depth": 5,
    "heads": 8,
    "mlp_dim": 512,
    "num_classes": 1000,
    "t2t_layers": "((7, 4), (3, 2), (3, 2))"
}

SimpleViT_parameters = {
    "image_size": 256,
    "patch_size": 32,
    "num_classes": 1000,
    "dim": 1024,
    "depth": 6,
    "heads": 16,
    "mlp_dim": 2048
}


DistillableViT = {
    "image_size": 256,
    "patch_size": 32,
    "num_classes": 1000,
    "dim": 1024,
    "depth": 6,
    "heads": 8,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1
}


DeepViT = {
    "image_size": 256,
    "patch_size": 32,
    "num_classes": 1000,
    "dim": 1024,
    "depth": 6,
    "heads": 16,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1
}


CaiT = {
    "image_size": 256,
    "patch_size": 32,
    "num_classes": 1000,
    "dim": 1024,
    "depth": 12,
    "cls_depth": 2,
    "heads": 16,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1,
    "layer_dropout": 0.05
}

ViT = {
    "image_size": 256,
    "patch_size": 32,
    "num_classes": 1000,
    "dim": 1024,
    "depth": 6,
    "heads": 16,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1

}

CustomViT = {
    "image_size": 128,
    "patch_sizes": [32, 16],
    "num_classes": 1000,
    "dim": 1024,
    "depth": 6,
    "heads": 16,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1

}


def get_default_parameters(model_name):
    """
    Returns default parameters of an model

    Parameters
    ----------
    model_name : str
        name of desired model

    Returns
    -------
    dict
        it includes default settings of the model.
    """
    parameters = globals()[model_name]
    return parameters
