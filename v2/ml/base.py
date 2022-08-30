import torch

from v2.utils import MetaParent


class BaseModel(metaclass=MetaParent):
    pass


class TorchModel(BaseModel, torch.nn.Module):
    pass
