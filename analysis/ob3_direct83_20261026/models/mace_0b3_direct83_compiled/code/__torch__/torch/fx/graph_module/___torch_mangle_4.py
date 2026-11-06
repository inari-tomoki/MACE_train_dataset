class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_4.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_4.GraphModule,
    x1: Tensor,
    x2: Tensor,
    w: Tensor) -> Tensor:
    empty = torch.empty(annotate(List[int], []), dtype=None, layout=None, device=torch.device("cpu"))
    getattr_1 = torch.size(x1)
    getitem = torch.slice(getattr_1, 0, -1)
    expand = torch.expand(empty, getitem)
    getattr_2 = torch.size(x2)
    getitem_1 = torch.slice(getattr_2, 0, -1)
    expand_1 = torch.expand(empty, getitem_1)
    getattr_3 = torch.size(w)
    getitem_2 = torch.slice(getattr_3, 0, -1)
    expand_2 = torch.expand(empty, getitem_2)
    broadcast_tensors = torch.broadcast_tensors([expand, expand_1, expand_2])
    getitem_3 = broadcast_tensors[0]
    getattr_4 = torch.size(getitem_3)
    add = torch.add(getattr_4, [-1])
    broadcast_to = torch.broadcast_to(x1, add)
    add_1 = torch.add(getattr_4, [-1])
    broadcast_to_1 = torch.broadcast_to(x2, add_1)
    add_2 = torch.add(getattr_4, [-1])
    broadcast_to_2 = torch.broadcast_to(w, add_2)
    add_3 = torch.add(getattr_4, [2048])
    reshape = torch.reshape(broadcast_to, [-1, 128])
    reshape_1 = torch.reshape(broadcast_to_1, [-1, 16])
    getattr_5 = torch.size(reshape)
    getitem_4 = getattr_5[0]
    reshape_2 = torch.reshape(broadcast_to_2, [-1, 512])
    reshape_3 = torch.reshape(reshape, [getitem_4, 128, 1])
    _0 = torch.slice(reshape_1, 0, 0, 9223372036854775807)
    getitem_5 = torch.slice(_0, 1, 0, 1)
    reshape_4 = torch.reshape(getitem_5, [getitem_4, 1, 1])
    _1 = torch.slice(reshape_1, 0, 0, 9223372036854775807)
    getitem_6 = torch.slice(_1, 1, 1, 4)
    reshape_5 = torch.reshape(getitem_6, [getitem_4, 1, 3])
    _2 = torch.slice(reshape_1, 0, 0, 9223372036854775807)
    getitem_7 = torch.slice(_2, 1, 4, 9)
    reshape_6 = torch.reshape(getitem_7, [getitem_4, 1, 5])
    _3 = torch.slice(reshape_1, 0, 0, 9223372036854775807)
    getitem_8 = torch.slice(_3, 1, 9, 16)
    reshape_7 = torch.reshape(getitem_8, [getitem_4, 1, 7])
    _4 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_9 = torch.slice(_4, 1, 0, 128)
    reshape_8 = torch.reshape(getitem_9, [-1, 128, 1])
    reshape_9 = torch.reshape(reshape_3, [getitem_4, 128])
    reshape_10 = torch.reshape(reshape_4, [getitem_4, 1])
    einsum_1 = torch.einsum("ca,cab->cab", [reshape_9, reshape_8])
    einsum_2 = torch.einsum("cab,cb->ca", [einsum_1, reshape_10])
    reshape_11 = torch.reshape(einsum_2, [getitem_4, 128])
    _5 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_10 = torch.slice(_5, 1, 128, 256)
    reshape_12 = torch.reshape(getitem_10, [-1, 128, 1])
    reshape_13 = torch.reshape(reshape_3, [getitem_4, 128])
    einsum_4 = torch.einsum("ca,cab->cab", [reshape_13, reshape_12])
    einsum_5 = torch.einsum("dbc,dca->dba", [einsum_4, reshape_5])
    reshape_14 = torch.reshape(einsum_5, [getitem_4, 384])
    _6 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_11 = torch.slice(_6, 1, 256, 384)
    reshape_15 = torch.reshape(getitem_11, [-1, 128, 1])
    reshape_16 = torch.reshape(reshape_3, [getitem_4, 128])
    einsum_7 = torch.einsum("ca,cab->cab", [reshape_16, reshape_15])
    einsum_8 = torch.einsum("dbc,dca->dba", [einsum_7, reshape_6])
    reshape_17 = torch.reshape(einsum_8, [getitem_4, 640])
    _7 = torch.slice(reshape_2, 0, 0, 9223372036854775807)
    getitem_12 = torch.slice(_7, 1, 384, 512)
    reshape_18 = torch.reshape(getitem_12, [-1, 128, 1])
    reshape_19 = torch.reshape(reshape_3, [getitem_4, 128])
    einsum_10 = torch.einsum("ca,cab->cab", [reshape_19, reshape_18])
    einsum_11 = torch.einsum("dbc,dca->dba", [einsum_10, reshape_7])
    reshape_20 = torch.reshape(einsum_11, [getitem_4, 896])
    _8 = [reshape_11, reshape_14, reshape_17, reshape_20]
    cat = torch.cat(_8, 1)
    return torch.reshape(cat, add_3)
