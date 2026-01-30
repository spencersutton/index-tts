from torch import nn


class MyModel(nn.Module):
    from indextts.s2mel.modules.flow_matching import CFM
    from indextts.s2mel.modules.length_regulator import InterpolateRegulator

    cfm: CFM
    length_regulator: InterpolateRegulator
    gpt_layer: nn.Sequential

    def __init__(self, dim: int = 512, in_channels: int = 80) -> None:
        super().__init__()
        from indextts.s2mel.modules.flow_matching import CFM
        from indextts.s2mel.modules.length_regulator import InterpolateRegulator

        self.cfm = CFM(dim=dim, in_channels=in_channels)
        self.length_regulator = InterpolateRegulator(dim)
        self.gpt_layer = nn.Sequential(nn.Linear(1280, 256), nn.Linear(256, 128), nn.Linear(128, 1024))

    def enable_torch_compile(self) -> None:
        """Enable torch.compile optimization.

        This method applies torch.compile to the model for significant
        performance improvements during inference.
        """
        self.cfm.enable_torch_compile()
