class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_5.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_5.GraphModule,
    x: Tensor,
    w: Tensor,
    b: Tensor) -> Tensor:
    _0 = __torch__.torch.functional.___torch_mangle_2.tensordot
    getattr_1 = torch.size(x)
    getitem = torch.slice(getattr_1, 0, -1)
    add = torch.add(getitem, [2048])
    reshape = torch.reshape(x, [-1, 2048])
    getattr_2 = torch.size(reshape)
    getitem_1 = getattr_2[0]
    reshape_1 = torch.reshape(w, [-1, 65536])
    narrow = torch.narrow(reshape, -1, 0, 128)
    reshape_2 = torch.reshape(narrow, [getitem_1, 128, 1])
    narrow_1 = torch.narrow(reshape, -1, 128, 384)
    reshape_3 = torch.reshape(narrow_1, [getitem_1, 128, 3])
    narrow_2 = torch.narrow(reshape, -1, 512, 640)
    reshape_4 = torch.reshape(narrow_2, [getitem_1, 128, 5])
    narrow_3 = torch.narrow(reshape, -1, 1152, 896)
    reshape_5 = torch.reshape(narrow_3, [getitem_1, 128, 7])
    narrow_4 = torch.narrow(reshape_1, -1, 0, 16384)
    reshape_6 = torch.reshape(narrow_4, [128, 128])
    mul = torch.mul(reshape_2, 0.088388347648318447)
    tensordot = _0(mul, reshape_6, ([1], [0]), None, )
    permute = torch.permute(tensordot, [0, 2, 1])
    reshape_7 = torch.reshape(permute, [getitem_1, 128])
    narrow_5 = torch.narrow(reshape_1, -1, 16384, 16384)
    reshape_8 = torch.reshape(narrow_5, [128, 128])
    mul_8 = torch.mul(reshape_3, 0.088388347648318447)
    tensordot_1 = _0(mul_8, reshape_8, ([1], [0]), None, )
    permute_1 = torch.permute(tensordot_1, [0, 2, 1])
    reshape_9 = torch.reshape(permute_1, [getitem_1, 384])
    narrow_6 = torch.narrow(reshape_1, -1, 32768, 16384)
    reshape_10 = torch.reshape(narrow_6, [128, 128])
    mul_9 = torch.mul(reshape_4, 0.088388347648318447)
    tensordot_2 = _0(mul_9, reshape_10, ([1], [0]), None, )
    permute_2 = torch.permute(tensordot_2, [0, 2, 1])
    reshape_11 = torch.reshape(permute_2, [getitem_1, 640])
    narrow_7 = torch.narrow(reshape_1, -1, 49152, 16384)
    reshape_12 = torch.reshape(narrow_7, [128, 128])
    mul_10 = torch.mul(reshape_5, 0.088388347648318447)
    tensordot_3 = _0(mul_10, reshape_12, ([1], [0]), None, )
    permute_3 = torch.permute(tensordot_3, [0, 2, 1])
    reshape_13 = torch.reshape(permute_3, [getitem_1, 896])
    _1 = [reshape_7, reshape_9, reshape_11, reshape_13]
    cat = torch.cat(_1, -1)
    return torch.reshape(cat, add)
