import torch


class ModelCombiner(torch.nn.Module):

    def __init__(self, models, n_class=1000) -> None:
        super().__init__()
        self.combined_models = torch.nn.ModuleList(modules=[])
        self.out_features = 0

        """
        for model in models:
            self.combined_models.append(model)"""

        for model in models:
            names = []
            for name, layer in model.named_parameters():
                name = name.split(".")[0]
                if name not in names:
                    names.append(name)
            last_layer = names[-1]
            self.out_features += getattr(model, last_layer)[-1].out_features

            setattr(model, last_layer, torch.nn.Sequential())

        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.out_features),
            torch.nn.Linear(self.out_features, n_class)
        )

    def forward(self, input):

        out = []

        for model in self.combined_models:
            print('input shape', input.shape)
            x = model(input)
            print("shape", x.shape)
            out.append(x)
        out = torch.cat(out, dim=1)
        out = self.mlp_head(out)
        print("mlp", out.shape)
        return out  # self.mlp(out)
