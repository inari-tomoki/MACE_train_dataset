class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  __annotations__["0"] = __torch__.torch.fx.graph_module.___torch_mangle_50.GraphModule
  __annotations__["1"] = __torch__.torch.fx.graph_module.___torch_mangle_51.GraphModule
  def __len__(self: __torch__.torch.nn.modules.container.___torch_mangle_55.ModuleList) -> int:
    return 2
