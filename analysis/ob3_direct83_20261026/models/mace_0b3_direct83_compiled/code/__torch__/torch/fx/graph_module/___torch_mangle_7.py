class tp_forward(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_7.tp_forward) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_7.tp_forward,
    x2: Tensor,
    w: Tensor) -> NoneType:
    _0 = "AssertionError: `right` method is not compiled, set `compile_right` to True when creating the TensorProduct"
    _1 = uninitialized(NoneType)
    ops.prim.RaiseException(_0)
    return _1
