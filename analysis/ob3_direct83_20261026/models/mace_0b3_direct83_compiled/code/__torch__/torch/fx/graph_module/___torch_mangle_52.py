class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_52.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_52.GraphModule,
    x: Tensor,
    y: Tensor) -> Tensor:
    einsum = torch.einsum("abc,abdec->abde", [y, x])
    return einsum
