def scatter_sum(src: Tensor,
    index: Tensor,
    dim: int=-1,
    out: Optional[Tensor]=None,
    dim_size: Optional[int]=None,
    reduce: str="sum") -> Tensor:
  _0 = __torch__.mace.tools.scatter._broadcast
  if torch.eq(reduce, "sum"):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  index0 = _0(index, src, dim, )
  if torch.__is__(out, None):
    size = torch.list(torch.size(src))
    if torch.__isnot__(dim_size, None):
      dim_size0 = unchecked_cast(int, dim_size)
      _2 = torch._set_item(size, dim, dim_size0)
    else:
      if torch.eq(torch.numel(index0), 0):
        _3 = torch._set_item(size, dim, 0)
      else:
        _4 = torch.add(int(torch.max(index0)), 1)
        _5 = torch._set_item(size, dim, _4)
    out0 = torch.zeros(size, dtype=ops.prim.dtype(src), layout=None, device=ops.prim.device(src))
    _6 = torch.scatter_add_(out0, dim, index0, src)
    _1 = _6
  else:
    out1 = unchecked_cast(Tensor, out)
    _7 = torch.scatter_add_(out1, dim, index0, src)
    _1 = _7
  return _1
def _broadcast(src: Tensor,
    other: Tensor,
    dim: int) -> Tensor:
  if torch.lt(dim, 0):
    dim0 = torch.add(torch.dim(other), dim)
  else:
    dim0 = dim
  if torch.eq(torch.dim(src), 1):
    src1 = src
    for _8 in range(torch.__range_length(0, dim0, 1)):
      src1 = torch.unsqueeze(src1, 0)
    src0 = src1
  else:
    src0 = src
  _9 = torch.__range_length(torch.dim(src0), torch.dim(other), 1)
  src2 = src0
  for _10 in range(_9):
    src2 = torch.unsqueeze(src2, -1)
  return torch.expand_as(src2, other)
