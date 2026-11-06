class GraphModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _code : str
  _lineno_map : Dict[int, int]
  _tracer_cls : NoneType
  _replace_hook : NoneType
  def __code_getter(self: __torch__.torch.fx.graph_module.___torch_mangle_8.GraphModule) -> str:
    return self._code
  def forward(self: __torch__.torch.fx.graph_module.___torch_mangle_8.GraphModule,
    x: Tensor,
    w: Tensor,
    b: Tensor) -> Tensor:
    _0 = __torch__.torch.functional.___torch_mangle_9.tensordot
    getattr_1 = torch.size(x)
    getitem = torch.slice(getattr_1, 0, -1)
    add = torch.add(getitem, [512])
    reshape = torch.reshape(x, [-1, 512])
    getattr_2 = torch.size(reshape)
    getitem_1 = getattr_2[0]
    reshape_1 = torch.reshape(w, [-1, 32768])
    narrow = torch.narrow(reshape, -1, 0, 128)
    reshape_2 = torch.reshape(narrow, [getitem_1, 128, 1])
    narrow_1 = torch.narrow(reshape, -1, 128, 384)
    reshape_3 = torch.reshape(narrow_1, [getitem_1, 128, 3])
    narrow_2 = torch.narrow(reshape_1, -1, 0, 16384)
    reshape_4 = torch.reshape(narrow_2, [128, 128])
    mul = torch.mul(reshape_2, 0.088388347648318447)
    tensordot = _0(mul, reshape_4, ([1], [0]), None, )
    permute = torch.permute(tensordot, [0, 2, 1])
    reshape_5 = torch.reshape(permute, [getitem_1, 128])
    narrow_3 = torch.narrow(reshape_1, -1, 16384, 16384)
    reshape_6 = torch.reshape(narrow_3, [128, 128])
    mul_4 = torch.mul(reshape_3, 0.088388347648318447)
    tensordot_1 = _0(mul_4, reshape_6, ([1], [0]), None, )
    permute_1 = torch.permute(tensordot_1, [0, 2, 1])
    reshape_7 = torch.reshape(permute_1, [getitem_1, 384])
    cat = torch.cat([reshape_5, reshape_7], -1)
    return torch.reshape(cat, add)
