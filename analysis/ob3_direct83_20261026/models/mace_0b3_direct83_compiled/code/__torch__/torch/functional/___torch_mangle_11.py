def tensordot(a: Tensor,
    b: Tensor,
    dims: Tuple[List[int], List[int]],
    out: Optional[Tensor]=None) -> Tensor:
  dims0 = unchecked_cast(Tuple[List[int], List[int]], dims)
  dims1 = unchecked_cast(Tuple[List[int], List[int]], dims0)
  dims_a, dims_b, = dims1
  if torch.__is__(out, None):
    _1 = torch.tensordot(a, b, dims_a, dims_b)
    _0 = _1
  else:
    out0 = unchecked_cast(Tensor, out)
    _2 = torch.tensordot(a, b, dims_a, dims_b, out=out0)
    _0 = _2
  return _0
