from ..utils import is_torch_available

"SpQR (Sparse-Quantized Representation) integration file"
if is_torch_available(): ...

def replace_with_spqr_linear(
    model, quantization_config=..., modules_to_not_convert=..., current_key_name=..., has_been_replaced=...
):  # -> tuple[Any, bool]:

    ...
