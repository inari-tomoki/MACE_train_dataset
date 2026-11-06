class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  __annotations__["0"] = __torch__.mace.modules.symmetric_contraction.___torch_mangle_67.Contraction
  def __len__(self: __torch__.torch.nn.modules.container.___torch_mangle_68.ModuleList) -> int:
    return 1
