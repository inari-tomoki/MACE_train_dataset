class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  __annotations__["0"] = __torch__.torch.fx.graph_module.___torch_mangle_44.GraphModule
  __annotations__["1"] = __torch__.torch.fx.graph_module.___torch_mangle_45.GraphModule
  def __len__(self: __torch__.torch.nn.modules.container.ModuleList) -> int:
    return 2
class ParameterList(Module):
  __parameters__ = ["0", "1", ]
  __buffers__ = []
  __annotations__ = []
  __annotations__["0"] = Tensor
  __annotations__["1"] = Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  _size : int
