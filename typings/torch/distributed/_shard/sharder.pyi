import abc
import torch.nn as nn

class Sharder(abc.ABC):
    @abc.abstractmethod
    def shard(self, module: nn.Module) -> nn.Module: ...
