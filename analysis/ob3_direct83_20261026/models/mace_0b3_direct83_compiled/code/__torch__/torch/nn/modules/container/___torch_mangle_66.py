class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  __annotations__["0"] = __torch__.torch.fx.graph_module.___torch_mangle_62.GraphModule
  __annotations__["1"] = __torch__.torch.fx.graph_module.___torch_mangle_63.GraphModule
  def __len__(self: __torch__.torch.nn.modules.container.___torch_mangle_66.ModuleList) -> int:
    return 2
