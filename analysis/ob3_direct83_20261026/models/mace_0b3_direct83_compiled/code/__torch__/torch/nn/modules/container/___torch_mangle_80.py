class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  __annotations__["0"] = __torch__.mace.modules.blocks.RealAgnosticDensityInteractionBlock
  __annotations__["1"] = __torch__.mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock
  def __len__(self: __torch__.torch.nn.modules.container.___torch_mangle_80.ModuleList) -> int:
    return 2
