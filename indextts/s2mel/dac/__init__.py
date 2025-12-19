__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["dac.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn as nn
from . import model as model
from . import utils as utils
from .model import DAC as DAC
from .model import DACFile as DACFile
