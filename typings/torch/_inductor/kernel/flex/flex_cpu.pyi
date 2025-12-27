"""CPU-specific implementations for flex attention"""

def check_cpu_supported(): ...
def lower_cpu(
    query, key, value, subgraph, block_mask, scale, kernel_options, score_mod_other_buffers, mask_mod_other_buffers
):
    """CPP based template for flex attention for x86 CPUs"""
