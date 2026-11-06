class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  __annotations__["0"] = __torch__.mace.modules.symmetric_contraction.Contraction
  __annotations__["1"] = __torch__.mace.modules.symmetric_contraction.___torch_mangle_57.Contraction
  def __len__(self: __torch__.torch.nn.modules.container.___torch_mangle_58.ModuleList) -> int:
    return 2
