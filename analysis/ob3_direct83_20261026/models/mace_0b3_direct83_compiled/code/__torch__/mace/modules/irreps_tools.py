class reshape_irreps(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  cueq_config : NoneType
  dims : List[int]
  muls : List[int]
  def forward(self: __torch__.mace.modules.irreps_tools.reshape_irreps,
    tensor: Tensor) -> Tensor:
    _0 = annotate(List[Tensor], [])
    batch, _1, = torch.size(tensor)
    muls = self.muls
    dims = self.dims
    _2 = ops.prim.min([torch.len(muls), torch.len(dims)])
    ix = 0
    for _3 in range(_2):
      mul = muls[_3]
      d = dims[_3]
      field = torch.slice(torch.slice(tensor), 1, ix, torch.add(ix, torch.mul(mul, d)))
      ix0 = torch.add(ix, torch.mul(mul, d))
      field0 = torch.reshape(field, [batch, mul, d])
      _4 = torch.append(_0, field0)
      ix = ix0
    return torch.cat(_0, -1)
def mask_head(x: Tensor,
    head: Tensor,
    num_heads: int) -> Tensor:
  _5 = (torch.size(x))[0]
  _6 = torch.floordiv((torch.size(x))[1], num_heads)
  _7 = ops.prim.device(x)
  mask = torch.zeros([_5, _6, num_heads], dtype=None, layout=None, device=_7)
  idx = torch.arange((torch.size(mask))[0], dtype=None, layout=None, device=ops.prim.device(x))
  _8 = torch.slice(mask, 1)
  _9 = torch.tensor(1, dtype=ops.prim.dtype(_8), device=ops.prim.device(_8))
  _10 = annotate(List[Optional[Tensor]], [idx, None, head])
  _11 = torch.index_put_(_8, _10, _9)
  mask0 = torch.reshape(torch.permute(mask, [0, 2, 1]), torch.size(x))
  return torch.mul(x, mask0)
