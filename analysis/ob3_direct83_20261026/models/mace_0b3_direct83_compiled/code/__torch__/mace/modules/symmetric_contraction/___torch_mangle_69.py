class SymmetricContraction(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  irreps_in : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  irreps_out : Tuple[Tuple[int, Tuple[int, int]]]
  internal_weights : bool
  shared_weights : NoneType
  contractions : __torch__.torch.nn.modules.container.___torch_mangle_68.ModuleList
  def forward(self: __torch__.mace.modules.symmetric_contraction.___torch_mangle_69.SymmetricContraction,
    x: Tensor,
    y: Tensor) -> Tensor:
    outs = annotate(List[Tensor], [])
    contractions = self.contractions
    _0 = getattr(contractions, "0")
    _1 = torch.append(outs, (_0).forward(x, y, ))
    return torch.cat(outs, -1)
