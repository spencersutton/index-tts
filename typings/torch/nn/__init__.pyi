from torch.nn import (
    attention as attention,
)
from torch.nn import (
    functional as functional,
)
from torch.nn import (
    init as init,
)
from torch.nn import (
    modules as modules,
)
from torch.nn import (
    parallel as parallel,
)
from torch.nn import (
    parameter as parameter,
)
from torch.nn import (
    utils as utils,
)
from torch.nn.modules import *
from torch.nn.parallel import DataParallel as DataParallel
from torch.nn.parameter import (
    Buffer as Buffer,
)
from torch.nn.parameter import (
    Parameter as Parameter,
)
from torch.nn.parameter import (
    UninitializedBuffer as UninitializedBuffer,
)
from torch.nn.parameter import (
    UninitializedParameter as UninitializedParameter,
)

def factory_kwargs(kwargs) -> dict[Any, Any]: ...
