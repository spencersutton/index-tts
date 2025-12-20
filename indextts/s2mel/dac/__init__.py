__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["dac.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import model as model, nn as nn, utils as utils
from .model import DAC as DAC, DACFile as DACFile
