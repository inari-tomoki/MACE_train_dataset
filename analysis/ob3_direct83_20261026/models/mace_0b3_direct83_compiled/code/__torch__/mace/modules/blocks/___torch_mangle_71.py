class EquivariantProductBasisBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  use_sc : bool
  use_agnostic_product : bool
  cueq_config : NoneType
  symmetric_contractions : __torch__.mace.modules.symmetric_contraction.___torch_mangle_69.SymmetricContraction
  linear : __torch__.e3nn.o3._linear.___torch_mangle_70.Linear
  def forward(self: __torch__.mace.modules.blocks.___torch_mangle_71.EquivariantProductBasisBlock,
    node_feats: Tensor,
    sc: Optional[Tensor],
    node_attrs: Tensor) -> Tensor:
    use_agnostic_product = self.use_agnostic_product
    if use_agnostic_product:
      _0 = (torch.size(node_feats))[0]
      _1 = ops.prim.dtype(node_feats)
      _2 = ops.prim.device(node_feats)
      node_attrs1 = torch.ones([_0, 1], dtype=_1, layout=None, device=_2)
      node_attrs0 = node_attrs1
    else:
      node_attrs0 = node_attrs
    symmetric_contractions = self.symmetric_contractions
    node_feats0 = (symmetric_contractions).forward(node_feats, node_attrs0, )
    use_sc = self.use_sc
    if use_sc:
      _3 = torch.__isnot__(sc, None)
    else:
      _3 = False
    if _3:
      sc0 = unchecked_cast(Tensor, sc)
      linear = self.linear
      _5 = (linear).forward(node_feats0, None, None, )
      _4 = torch.add(_5, sc0)
    else:
      linear0 = self.linear
      _6 = (linear0).forward(node_feats0, None, None, )
      _4 = _6
    return _4
