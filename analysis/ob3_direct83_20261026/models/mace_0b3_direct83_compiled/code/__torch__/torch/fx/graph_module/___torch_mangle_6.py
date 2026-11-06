class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_6.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_6.GraphModule,
    x1: Tensor,
    x2: Tensor,
    w: Tensor) -> Tensor:
    _0 = __torch__.torch.functional.___torch_mangle_3.tensordot
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
    add_2 = torch.add(getattr_3, [2048])
    reshape = torch.reshape(broadcast_to, [-1, 2048])
    reshape_1 = torch.reshape(broadcast_to_1, [-1, 10])
    getattr_4 = torch.size(reshape)
    getitem_3 = getattr_4[0]
    reshape_2 = torch.reshape(w, [-1, 655360])
    _1 = torch.slice(reshape, 0, 0, 9223372036854775807)
    getitem_4 = torch.slice(_1, 1, 0, 128)
    reshape_3 = torch.reshape(getitem_4, [getitem_3, 128, 1])
    _2 = torch.slice(reshape, 0, 0, 9223372036854775807)
    getitem_5 = torch.slice(_2, 1, 128, 512)
    reshape_4 = torch.reshape(getitem_5, [getitem_3, 128, 3])
    _3 = torch.slice(reshape, 0, 0, 9223372036854775807)
    getitem_6 = torch.slice(_3, 1, 512, 1152)
    reshape_5 = torch.reshape(getitem_6, [getitem_3, 128, 5])
    _4 = torch.slice(reshape, 0, 0, 9223372036854775807)
    getitem_7 = torch.slice(_4, 1, 1152, 2048)
    reshape_6 = torch.reshape(getitem_7, [getitem_3, 128, 7])
    reshape_7 = torch.reshape(reshape_1, [getitem_3, 10, 1])
    _5 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_8 = torch.slice(_5, 1, 0, 163840)
    reshape_8 = torch.reshape(getitem_8, [128, 10, 128])
    reshape_9 = torch.reshape(reshape_3, [getitem_3, 128])
    reshape_10 = torch.reshape(reshape_7, [getitem_3, 10])
    mul = torch.mul(reshape_10, 0.02795084971874737)
    einsum_1 = torch.einsum("cb,ca->cba", [mul, reshape_9])
    tensordot = _0(einsum_1, reshape_8, ([1, 2], [1, 0]), None, )
    reshape_11 = torch.reshape(tensordot, [getitem_3, 128])
    _6 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_9 = torch.slice(_6, 1, 163840, 327680)
    reshape_12 = torch.reshape(getitem_9, [128, 10, 128])
    reshape_13 = torch.reshape(reshape_7, [getitem_3, 10])
    mul_8 = torch.mul(reshape_13, 0.027950849718747377)
    tensordot_1 = _0(mul_8, reshape_12, ([1], [1]), None, )
    einsum_3 = torch.einsum("dbc,dba->dca", [tensordot_1, reshape_4])
    reshape_14 = torch.reshape(einsum_3, [getitem_3, 384])
    _7 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_10 = torch.slice(_7, 1, 327680, 491520)
    reshape_15 = torch.reshape(getitem_10, [128, 10, 128])
    reshape_16 = torch.reshape(reshape_7, [getitem_3, 10])
    mul_9 = torch.mul(reshape_16, 0.02795084971874737)
    tensordot_2 = _0(mul_9, reshape_15, ([1], [1]), None, )
    einsum_5 = torch.einsum("dbc,dba->dca", [tensordot_2, reshape_5])
    reshape_17 = torch.reshape(einsum_5, [getitem_3, 640])
    _8 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_11 = torch.slice(_8, 1, 491520, 655360)
    reshape_18 = torch.reshape(getitem_11, [128, 10, 128])
    reshape_19 = torch.reshape(reshape_7, [getitem_3, 10])
    mul_10 = torch.mul(reshape_19, 0.027950849718747367)
    tensordot_3 = _0(mul_10, reshape_18, ([1], [1]), None, )
    einsum_7 = torch.einsum("dbc,dba->dca", [tensordot_3, reshape_6])
    reshape_20 = torch.reshape(einsum_7, [getitem_3, 896])
    _9 = [reshape_11, reshape_14, reshape_17, reshape_20]
    cat = torch.cat(_9, 1)
    return torch.reshape(cat, add_2)
