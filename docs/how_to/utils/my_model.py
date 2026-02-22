import torch


class MyEncoder(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Storing constructor args as same-named attributes
        # allows TorchModelWrapper to auto-infer them for serialization.
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)
