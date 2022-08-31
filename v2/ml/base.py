import os
import torch

from v2.utils import MetaParent


class BaseModel(metaclass=MetaParent):
    pass


class TorchModel(BaseModel, torch.nn.Module):

    def save(self, dir, filename):
        os.makedirs(dir, exist_ok=True)
        return torch.save(self.state_dict(), os.path.join(dir, filename))

    def restore(self, filepath):
        return self.load_state_dict(torch.load(filepath))
