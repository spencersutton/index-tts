from collections.abc import Sized

import torch

class Tensor(torch._C.TensorBase, Sized): ...
