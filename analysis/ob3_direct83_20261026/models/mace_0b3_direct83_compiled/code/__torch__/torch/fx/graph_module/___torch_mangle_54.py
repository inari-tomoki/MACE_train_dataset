class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_54.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_54.GraphModule,
    x: Tensor,
    y: Tensor,
    w: Tensor,
    z: Tensor) -> Tensor:
    _0 = __torch__.torch.functional.___torch_mangle_8.tensordot
    tensordot = _0(z, y, ([1], [0]), None, )
    einsum = torch.einsum("adb,abc->adbc", [tensordot, w])
    tensordot_1 = _0(einsum, x, ([1, 3], [4, 3]), None, )
    return tensordot_1
