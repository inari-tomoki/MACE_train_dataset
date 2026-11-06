class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_17.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_17.GraphModule,
    x: Tensor,
    w: Tensor,
    b: Tensor) -> Tensor:
    _0 = __torch__.torch.functional.___torch_mangle_12.tensordot
    getattr_1 = torch.size(x)
    getitem = torch.slice(getattr_1, 0, -1)
    add = torch.add(getitem, [16])
    reshape = torch.reshape(x, [-1, 128])
    getattr_2 = torch.size(reshape)
    getitem_1 = getattr_2[0]
    reshape_1 = torch.reshape(w, [-1, 2048])
    reshape_2 = torch.reshape(reshape, [getitem_1, 128, 1])
    reshape_3 = torch.reshape(reshape_1, [128, 16])
    tensordot = _0(reshape_2, reshape_3, ([1], [0]), None, )
    mul = torch.mul(tensordot, 0.088388347648318447)
    permute = torch.permute(mul, [0, 2, 1])
    reshape_4 = torch.reshape(permute, [getitem_1, 16])
    return torch.reshape(reshape_4, add)
