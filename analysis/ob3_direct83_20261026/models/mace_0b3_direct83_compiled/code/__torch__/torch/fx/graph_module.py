class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.GraphModule,
    x: Tensor,
    w: Tensor,
    b: Tensor) -> Tensor:
    _0 = __torch__.torch.functional.___torch_mangle_0.tensordot
    getattr_1 = torch.size(x)
    getitem = torch.slice(getattr_1, 0, -1)
    add = torch.add(getitem, [128])
    reshape = torch.reshape(x, [-1, 10])
    getattr_2 = torch.size(reshape)
    getitem_1 = getattr_2[0]
    reshape_1 = torch.reshape(w, [-1, 1280])
    reshape_2 = torch.reshape(reshape, [getitem_1, 10, 1])
    reshape_3 = torch.reshape(reshape_1, [10, 128])
    mul = torch.mul(reshape_2, 0.31622776601683794)
    tensordot = _0(mul, reshape_3, ([1], [0]), None, )
    permute = torch.permute(tensordot, [0, 2, 1])
    reshape_4 = torch.reshape(permute, [getitem_1, 128])
    return torch.reshape(reshape_4, add)
class tp_forward(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.tp_forward) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.tp_forward,
    x2: Tensor,
    w: Tensor) -> NoneType:
    _1 = "AssertionError: `right` method is not compiled, set `compile_right` to True when creating the TensorProduct"
    _2 = uninitialized(NoneType)
    ops.prim.RaiseException(_1)
    return _2
