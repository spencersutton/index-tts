from transformers.utils.quantization_config import FPQuantConfig

from ..utils import is_fp_quant_available

"FP-Quant integration file"
if is_fp_quant_available(): ...

def adapt_fp_quant_config(config: FPQuantConfig): ...
