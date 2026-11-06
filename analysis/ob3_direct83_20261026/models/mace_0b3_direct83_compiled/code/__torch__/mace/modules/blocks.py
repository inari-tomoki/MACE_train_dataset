class LinearNodeEmbeddingBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  linear : __torch__.e3nn.o3._linear.Linear
  def forward(self: __torch__.mace.modules.blocks.LinearNodeEmbeddingBlock,
    node_attrs: Tensor) -> Tensor:
    linear = self.linear
    _0 = (linear).forward(node_attrs, None, None, )
    return _0
class RadialEmbeddingBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  out_dim : int
  apply_cutoff : bool
  bessel_fn : __torch__.mace.modules.radial.BesselBasis
  distance_transform : __torch__.mace.modules.radial.AgnesiTransform
  cutoff_fn : __torch__.mace.modules.radial.PolynomialCutoff
  def forward(self: __torch__.mace.modules.blocks.RadialEmbeddingBlock,
    edge_lengths: Tensor,
    node_attrs: Tensor,
    edge_index: Tensor,
    atomic_numbers: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    cutoff_fn = self.cutoff_fn
    cutoff = (cutoff_fn).forward(edge_lengths, )
    distance_transform = self.distance_transform
    edge_lengths0 = (distance_transform).forward(edge_lengths, node_attrs, edge_index, atomic_numbers, )
    bessel_fn = self.bessel_fn
    radial = (bessel_fn).forward(edge_lengths0, )
    apply_cutoff = self.apply_cutoff
    if torch.__not__(apply_cutoff):
      _1 = (radial, cutoff)
    else:
      _1 = (torch.mul(radial, cutoff), None)
    return _1
class AtomicEnergiesBlock(Module):
  __parameters__ = []
  __buffers__ = ["atomic_energies", ]
  atomic_energies : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  def forward(self: __torch__.mace.modules.blocks.AtomicEnergiesBlock,
    x: Tensor) -> Tensor:
    atomic_energies = self.atomic_energies
    _2 = torch.numpy_T(torch.atleast_2d(atomic_energies))
    _3 = ops.prim.dtype(x)
    energies = torch.to(_2, ops.prim.device(x), _3)
    return torch.matmul(x, energies)
class RealAgnosticDensityInteractionBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  node_attrs_irreps : Tuple[Tuple[int, Tuple[int, int]]]
  node_feats_irreps : Tuple[Tuple[int, Tuple[int, int]]]
  edge_attrs_irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  edge_feats_irreps : Tuple[Tuple[int, Tuple[int, int]]]
  target_irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  hidden_irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  avg_num_neighbors : float
  radial_MLP : List[int]
  edge_irreps : Tuple[Tuple[int, Tuple[int, int]]]
  cueq_config : NoneType
  oeq_config : NoneType
  irreps_out : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  linear_up : __torch__.e3nn.o3._linear.___torch_mangle_19.Linear
  conv_tp : __torch__.e3nn.o3._tensor_product._tensor_product.TensorProduct
  conv_tp_weights : __torch__.e3nn.nn._fc.FullyConnectedNet
  linear : __torch__.e3nn.o3._linear.___torch_mangle_28.Linear
  skip_tp : __torch__.e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct
  density_fn : __torch__.e3nn.nn._fc.___torch_mangle_29.FullyConnectedNet
  reshape : __torch__.mace.modules.irreps_tools.reshape_irreps
  def forward(self: __torch__.mace.modules.blocks.RealAgnosticDensityInteractionBlock,
    node_attrs: Tensor,
    node_feats: Tensor,
    edge_attrs: Tensor,
    edge_feats: Tensor,
    edge_index: Tensor,
    cutoff: Optional[Tensor]=None,
    lammps_class: Optional[Any]=None,
    lammps_natoms: Tuple[int, int]=(0, 0),
    first_layer: bool=False) -> Tuple[Tensor, NoneType]:
    _4 = __torch__.mace.tools.scatter.scatter_sum
    receiver = torch.select(edge_index, 0, 1)
    num_nodes = (torch.size(node_feats))[0]
    if torch.__isnot__(lammps_class, None):
      lammps_class0, n_real = unchecked_cast(Any, lammps_class), (lammps_natoms)[0]
    else:
      lammps_class0, n_real = lammps_class, None
    linear_up = self.linear_up
    node_feats0 = (linear_up).forward(node_feats, None, None, )
    node_feats1 = (self).handle_lammps(node_feats0, lammps_class0, lammps_natoms, first_layer, )
    conv_tp_weights = self.conv_tp_weights
    tp_weights = (conv_tp_weights).forward(edge_feats, )
    density_fn = self.density_fn
    _5 = torch.pow((density_fn).forward(edge_feats, ), 2)
    edge_density = torch.tanh(_5)
    if torch.__isnot__(cutoff, None):
      cutoff0 = unchecked_cast(Tensor, cutoff)
      tp_weights1 = torch.mul(tp_weights, cutoff0)
      edge_density0, tp_weights0 = torch.mul(edge_density, cutoff0), tp_weights1
    else:
      edge_density0, tp_weights0 = edge_density, tp_weights
    density = _4(edge_density0, receiver, 0, None, num_nodes, "sum", )
    conv_tp = self.conv_tp
    _6 = annotate(List[Optional[Tensor]], [torch.select(edge_index, 0, 0)])
    mji = (conv_tp).forward(torch.index(node_feats1, _6), edge_attrs, tp_weights0, )
    message = _4(mji, torch.select(edge_index, 0, 1), 0, None, (torch.size(node_feats1))[0], "sum", )
    message0 = (self).truncate_ghosts(message, n_real, )
    node_attrs0 = (self).truncate_ghosts(node_attrs, n_real, )
    density0 = (self).truncate_ghosts(density, n_real, )
    linear = self.linear
    _7 = (linear).forward(message0, None, None, )
    message1 = torch.div(_7, torch.add(density0, 1))
    skip_tp = self.skip_tp
    message2 = (skip_tp).forward(message1, node_attrs0, None, )
    reshape = self.reshape
    _8 = ((reshape).forward(message2, ), None)
    return _8
  def handle_lammps(self: __torch__.mace.modules.blocks.RealAgnosticDensityInteractionBlock,
    node_feats: Tensor,
    lammps_class: Optional[Any],
    lammps_natoms: Tuple[int, int],
    first_layer: bool) -> Tensor:
    return node_feats
  def truncate_ghosts(self: __torch__.mace.modules.blocks.RealAgnosticDensityInteractionBlock,
    tensor: Tensor,
    n_real: Optional[int]=None) -> Tensor:
    if torch.__isnot__(n_real, None):
      n_real0 = unchecked_cast(int, n_real)
      _10 = torch.slice(tensor, 0, None, n_real0)
      _9 = _10
    else:
      _9 = tensor
    return _9
class RealAgnosticDensityResidualInteractionBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  node_attrs_irreps : Tuple[Tuple[int, Tuple[int, int]]]
  node_feats_irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  edge_attrs_irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  edge_feats_irreps : Tuple[Tuple[int, Tuple[int, int]]]
  target_irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  hidden_irreps : str
  avg_num_neighbors : float
  radial_MLP : List[int]
  edge_irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  cueq_config : NoneType
  oeq_config : NoneType
  irreps_out : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  linear_up : __torch__.e3nn.o3._linear.___torch_mangle_30.Linear
  conv_tp : __torch__.e3nn.o3._tensor_product._tensor_product.___torch_mangle_31.TensorProduct
  conv_tp_weights : __torch__.e3nn.nn._fc.___torch_mangle_41.FullyConnectedNet
  linear : __torch__.e3nn.o3._linear.___torch_mangle_42.Linear
  skip_tp : __torch__.e3nn.o3._tensor_product._sub.___torch_mangle_43.FullyConnectedTensorProduct
  density_fn : __torch__.e3nn.nn._fc.___torch_mangle_29.FullyConnectedNet
  reshape : __torch__.mace.modules.irreps_tools.reshape_irreps
  def forward(self: __torch__.mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock,
    node_attrs: Tensor,
    node_feats: Tensor,
    edge_attrs: Tensor,
    edge_feats: Tensor,
    edge_index: Tensor,
    cutoff: Optional[Tensor]=None,
    lammps_class: Optional[Any]=None,
    lammps_natoms: Tuple[int, int]=(0, 0),
    first_layer: bool=False) -> Tuple[Tensor, Tensor]:
    _11 = __torch__.mace.tools.scatter.scatter_sum
    receiver = torch.select(edge_index, 0, 1)
    num_nodes = (torch.size(node_feats))[0]
    if torch.__isnot__(lammps_class, None):
      lammps_class1, n_real = unchecked_cast(Any, lammps_class), (lammps_natoms)[0]
    else:
      lammps_class1, n_real = lammps_class, None
    skip_tp = self.skip_tp
    sc = (skip_tp).forward(node_feats, node_attrs, None, )
    linear_up = self.linear_up
    node_feats2 = (linear_up).forward(node_feats, None, None, )
    node_feats3 = (self).handle_lammps(node_feats2, lammps_class1, lammps_natoms, first_layer, )
    conv_tp_weights = self.conv_tp_weights
    tp_weights = (conv_tp_weights).forward(edge_feats, )
    density_fn = self.density_fn
    _12 = torch.pow((density_fn).forward(edge_feats, ), 2)
    edge_density = torch.tanh(_12)
    if torch.__isnot__(cutoff, None):
      cutoff1 = unchecked_cast(Tensor, cutoff)
      tp_weights3 = torch.mul(tp_weights, cutoff1)
      edge_density1, tp_weights2 = torch.mul(edge_density, cutoff1), tp_weights3
    else:
      edge_density1, tp_weights2 = edge_density, tp_weights
    density = _11(edge_density1, receiver, 0, None, num_nodes, "sum", )
    conv_tp = self.conv_tp
    _13 = annotate(List[Optional[Tensor]], [torch.select(edge_index, 0, 0)])
    mji = (conv_tp).forward(torch.index(node_feats3, _13), edge_attrs, tp_weights2, )
    message = _11(mji, torch.select(edge_index, 0, 1), 0, None, (torch.size(node_feats3))[0], "sum", )
    message3 = (self).truncate_ghosts(message, n_real, )
    node_attrs1 = (self).truncate_ghosts(node_attrs, n_real, )
    density1 = (self).truncate_ghosts(density, n_real, )
    sc0 = (self).truncate_ghosts(sc, n_real, )
    linear = self.linear
    _14 = (linear).forward(message3, None, None, )
    message4 = torch.div(_14, torch.add(density1, 1))
    reshape = self.reshape
    _15 = ((reshape).forward(message4, ), sc0)
    return _15
  def handle_lammps(self: __torch__.mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock,
    node_feats: Tensor,
    lammps_class: Optional[Any],
    lammps_natoms: Tuple[int, int],
    first_layer: bool) -> Tensor:
    return node_feats
  def truncate_ghosts(self: __torch__.mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock,
    tensor: Tensor,
    n_real: Optional[int]=None) -> Tensor:
    if torch.__isnot__(n_real, None):
      n_real1 = unchecked_cast(int, n_real)
      _17 = torch.slice(tensor, 0, None, n_real1)
      _16 = _17
    else:
      _16 = tensor
    return _16
class EquivariantProductBasisBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  use_sc : bool
  use_agnostic_product : bool
  cueq_config : NoneType
  symmetric_contractions : __torch__.mace.modules.symmetric_contraction.SymmetricContraction
  linear : __torch__.e3nn.o3._linear.___torch_mangle_59.Linear
  def forward(self: __torch__.mace.modules.blocks.EquivariantProductBasisBlock,
    node_feats: Tensor,
    sc: Optional[Tensor],
    node_attrs: Tensor) -> Tensor:
    use_agnostic_product = self.use_agnostic_product
    if use_agnostic_product:
      _18 = (torch.size(node_feats))[0]
      _19 = ops.prim.dtype(node_feats)
      _20 = ops.prim.device(node_feats)
      node_attrs3 = torch.ones([_18, 1], dtype=_19, layout=None, device=_20)
      node_attrs2 = node_attrs3
    else:
      node_attrs2 = node_attrs
    symmetric_contractions = self.symmetric_contractions
    node_feats4 = (symmetric_contractions).forward(node_feats, node_attrs2, )
    use_sc = self.use_sc
    if use_sc:
      _21 = torch.__isnot__(sc, None)
    else:
      _21 = False
    if _21:
      sc1 = unchecked_cast(Tensor, sc)
      linear = self.linear
      _23 = (linear).forward(node_feats4, None, None, )
      _22 = torch.add(_23, sc1)
    else:
      linear0 = self.linear
      _24 = (linear0).forward(node_feats4, None, None, )
      _22 = _24
    return _22
class LinearReadoutBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  linear : __torch__.e3nn.o3._linear.___torch_mangle_72.Linear
  def forward(self: __torch__.mace.modules.blocks.LinearReadoutBlock,
    x: Tensor,
    heads: Optional[Tensor]=None) -> Tensor:
    linear = self.linear
    return (linear).forward(x, None, None, )
class NonLinearReadoutBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  hidden_irreps : Tuple[Tuple[int, Tuple[int, int]]]
  num_heads : int
  linear_1 : __torch__.e3nn.o3._linear.___torch_mangle_73.Linear
  non_linearity : __torch__.e3nn.nn._activation.Activation
  linear_2 : __torch__.e3nn.o3._linear.___torch_mangle_79.Linear
  def forward(self: __torch__.mace.modules.blocks.NonLinearReadoutBlock,
    x: Tensor,
    heads: Optional[Tensor]=None) -> Tensor:
    _25 = __torch__.mace.modules.irreps_tools.mask_head
    non_linearity = self.non_linearity
    linear_1 = self.linear_1
    x0 = (non_linearity).forward((linear_1).forward(x, None, None, ), )
    num_heads = self.num_heads
    if torch.gt(num_heads, 1):
      _26 = torch.__isnot__(heads, None)
    else:
      _26 = False
    if _26:
      heads0 = unchecked_cast(Tensor, heads)
      num_heads0 = self.num_heads
      x1 = _25(x0, heads0, num_heads0, )
    else:
      x1 = x0
    linear_2 = self.linear_2
    _27 = (linear_2).forward(x1, None, None, )
    return _27
class ScaleShiftBlock(Module):
  __parameters__ = []
  __buffers__ = ["scale", "shift", ]
  scale : Tensor
  shift : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.mace.modules.blocks.ScaleShiftBlock,
    x: Tensor,
    head: Tensor) -> Tensor:
    scale = self.scale
    _28 = torch.atleast_1d(scale)
    _29 = annotate(List[Optional[Tensor]], [head])
    _30 = torch.mul(torch.index(_28, _29), x)
    shift = self.shift
    _31 = torch.atleast_1d(shift)
    _32 = annotate(List[Optional[Tensor]], [head])
    _33 = torch.add(_30, torch.index(_31, _32))
    return _33
