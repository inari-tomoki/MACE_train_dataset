class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_14.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_14.GraphModule,
    x1: Tensor,
    x2: Tensor,
    w: Tensor) -> Tensor:
    _0 = __torch__.torch.functional.___torch_mangle_7.tensordot
    empty = torch.empty(annotate(List[int], []), dtype=None, layout=None, device=torch.device("cpu"))
    getattr_1 = torch.size(x1)
    getitem = torch.slice(getattr_1, 0, -1)
    expand = torch.expand(empty, getitem)
    getattr_2 = torch.size(x2)
    getitem_1 = torch.slice(getattr_2, 0, -1)
    expand_1 = torch.expand(empty, getitem_1)
    broadcast_tensors = torch.broadcast_tensors([expand, expand_1])
    getitem_2 = broadcast_tensors[0]
    getattr_3 = torch.size(getitem_2)
    add = torch.add(getattr_3, [-1])
    broadcast_to = torch.broadcast_to(x1, add)
    add_1 = torch.add(getattr_3, [-1])
    broadcast_to_1 = torch.broadcast_to(x2, add_1)
    add_2 = torch.add(getattr_3, [128])
    reshape = torch.reshape(broadcast_to, [-1, 512])
    reshape_1 = torch.reshape(broadcast_to_1, [-1, 10])
    getattr_4 = torch.size(reshape)
    getitem_3 = getattr_4[0]
    reshape_2 = torch.reshape(w, [-1, 163840])
    _1 = torch.slice(reshape, 0, 0, 9223372036854775807)
    getitem_4 = torch.slice(_1, 1, 0, 128)
    reshape_3 = torch.reshape(getitem_4, [getitem_3, 128, 1])
    reshape_5 = torch.reshape(reshape_1, [getitem_3, 10, 1])
    _2 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_6 = torch.slice(_2, 1, 0, 163840)
    reshape_6 = torch.reshape(getitem_6, [128, 10, 128])
    reshape_7 = torch.reshape(reshape_3, [getitem_3, 128])
    reshape_8 = torch.reshape(reshape_5, [getitem_3, 10])
    mul = torch.mul(reshape_8, 0.02795084971874737)
    einsum_1 = torch.einsum("cb,ca->cba", [mul, reshape_7])
    tensordot = _0(einsum_1, reshape_6, ([1, 2], [1, 0]), None, )
    reshape_9 = torch.reshape(tensordot, [getitem_3, 128])
    return torch.reshape(reshape_9, add_2)
