class ScaleShiftMACE(Module):
  __parameters__ = []
  __buffers__ = ["atomic_numbers", "r_max", "num_interactions", ]
  atomic_numbers : Tensor
  r_max : Tensor
  num_interactions : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  heads : List[str]
  lammps_mliap : bool
  apply_cutoff : bool
  edge_irreps : NoneType
  use_reduced_cg : bool
  use_agnostic_product : bool
  use_so3 : bool
  use_last_readout_only : bool
  use_edge_irreps_first : bool
  pair_repulsion : bool
  node_embedding : __torch__.mace.modules.blocks.LinearNodeEmbeddingBlock
  radial_embedding : __torch__.mace.modules.blocks.RadialEmbeddingBlock
  pair_repulsion_fn : __torch__.mace.modules.radial.ZBLBasis
  spherical_harmonics : __torch__.e3nn.o3._spherical_harmonics.SphericalHarmonics
  atomic_energies_fn : __torch__.mace.modules.blocks.AtomicEnergiesBlock
  interactions : __torch__.torch.nn.modules.container.___torch_mangle_80.ModuleList
  products : __torch__.torch.nn.modules.container.___torch_mangle_81.ModuleList
  readouts : __torch__.torch.nn.modules.container.___torch_mangle_82.ModuleList
  scale_shift : __torch__.mace.modules.blocks.ScaleShiftBlock
  def forward(self: __torch__.mace.modules.models.ScaleShiftMACE,
    data: Dict[str, Tensor],
    training: bool=False,
    compute_force: bool=True,
    compute_virials: bool=False,
    compute_stress: bool=False,
    compute_displacement: bool=False,
    compute_hessian: bool=False,
    compute_edge_forces: bool=False,
    compute_atomic_stresses: bool=False,
    lammps_mliap: bool=False) -> Dict[str, Optional[Tensor]]:
    _0 = __torch__.mace.modules.utils.prepare_graph
    _1 = __torch__.mace.tools.scatter.scatter_sum
    _2 = __torch__.mace.modules.utils.get_outputs
    _3 = __torch__.mace.modules.utils.get_atomic_virials_stresses
    ctx = _0(data, compute_virials, compute_stress, compute_displacement, lammps_mliap, )
    is_lammps = (ctx)[0]
    num_atoms_arange = torch.to((ctx)[2], 4)
    num_graphs = (ctx)[1]
    displacement = (ctx)[3]
    positions = (ctx)[4]
    vectors = (ctx)[5]
    lengths = (ctx)[6]
    cell = (ctx)[7]
    node_heads = torch.to((ctx)[8], 4)
    interaction_kwargs = (ctx)[9]
    lammps_natoms = (interaction_kwargs)[1]
    lammps_class = (interaction_kwargs)[0]
    atomic_energies_fn = self.atomic_energies_fn
    _4 = (atomic_energies_fn).forward(data["node_attrs"], )
    _5 = annotate(List[Optional[Tensor]], [num_atoms_arange, node_heads])
    node_e0 = torch.index(_4, _5)
    _6 = _1(node_e0, data["batch"], 0, None, num_graphs, "sum", )
    e0 = torch.to(_6, ops.prim.dtype(vectors))
    node_embedding = self.node_embedding
    node_feats = (node_embedding).forward(data["node_attrs"], )
    spherical_harmonics = self.spherical_harmonics
    edge_attrs = (spherical_harmonics).forward(vectors, )
    radial_embedding = self.radial_embedding
    _7 = data["node_attrs"]
    _8 = data["edge_index"]
    atomic_numbers = self.atomic_numbers
    _9 = (radial_embedding).forward(lengths, _7, _8, atomic_numbers, )
    edge_feats, cutoff, = _9
    pair_repulsion_fn = self.pair_repulsion_fn
    _10 = data["node_attrs"]
    _11 = data["edge_index"]
    atomic_numbers0 = self.atomic_numbers
    pair_node_energy = (pair_repulsion_fn).forward(lengths, _10, _11, atomic_numbers0, )
    if is_lammps:
      pair_node_energy1 = torch.slice(pair_node_energy, 0, None, (lammps_natoms)[0])
      pair_node_energy0 = pair_node_energy1
    else:
      pair_node_energy0 = pair_node_energy
    node_es_list = [pair_node_energy0]
    node_feats_list = annotate(List[Tensor], [])
    interactions = self.interactions
    _00 = getattr(interactions, "0")
    _12 = getattr(interactions, "1")
    products = self.products
    _01 = getattr(products, "0")
    _13 = getattr(products, "1")
    node_attrs_slice = data["node_attrs"]
    _14 = (_00).forward(node_attrs_slice, node_feats, edge_attrs, edge_feats, data["edge_index"], cutoff, lammps_class, lammps_natoms, True, )
    node_feats0, sc, = _14
    if is_lammps:
      _15 = True
    else:
      _15 = False
    if _15:
      node_attrs_slice1 = torch.slice(node_attrs_slice, 0, None, (lammps_natoms)[0])
      node_attrs_slice0 = node_attrs_slice1
    else:
      node_attrs_slice0 = node_attrs_slice
    node_feats1 = (_01).forward(node_feats0, sc, node_attrs_slice0, )
    _16 = torch.append(node_feats_list, node_feats1)
    node_attrs_slice2 = data["node_attrs"]
    if is_lammps:
      _17 = True
    else:
      _17 = False
    if _17:
      node_attrs_slice4 = torch.slice(node_attrs_slice2, 0, None, (lammps_natoms)[0])
      node_attrs_slice3 = node_attrs_slice4
    else:
      node_attrs_slice3 = node_attrs_slice2
    _18 = (_12).forward(node_attrs_slice3, node_feats1, edge_attrs, edge_feats, data["edge_index"], cutoff, lammps_class, lammps_natoms, False, )
    node_feats2, sc0, = _18
    node_feats3 = (_13).forward(node_feats2, sc0, node_attrs_slice3, )
    _19 = torch.append(node_feats_list, node_feats3)
    readouts = self.readouts
    _015 = getattr(readouts, "0")
    _116 = getattr(readouts, "1")
    readouts0 = self.readouts
    if torch.eq((readouts0).__len__(), 1):
      feat_idx = -1
    else:
      feat_idx = 0
    _20 = (_015).forward(node_feats_list[feat_idx], node_heads, )
    _21 = annotate(List[Optional[Tensor]], [num_atoms_arange, node_heads])
    _22 = torch.append(node_es_list, torch.index(_20, _21))
    readouts1 = self.readouts
    if torch.eq((readouts1).__len__(), 1):
      feat_idx0 = -1
    else:
      feat_idx0 = 1
    _23 = (_116).forward(node_feats_list[feat_idx0], node_heads, )
    _24 = annotate(List[Optional[Tensor]], [num_atoms_arange, node_heads])
    _25 = torch.append(node_es_list, torch.index(_23, _24))
    node_feats_out = torch.cat(node_feats_list, -1)
    node_inter_es = torch.sum(torch.stack(node_es_list), [0])
    scale_shift = self.scale_shift
    node_inter_es0 = (scale_shift).forward(node_inter_es, node_heads, )
    inter_e = _1(node_inter_es0, data["batch"], -1, None, num_graphs, "sum", )
    total_energy = torch.add(e0, inter_e)
    _26 = torch.to(torch.clone(node_e0), 7)
    _27 = torch.to(torch.clone(node_inter_es0), 7)
    node_energy = torch.add(_26, _27)
    if compute_edge_forces:
      _28 = True
    else:
      _28 = compute_atomic_stresses
    _29 = _2(inter_e, positions, cell, displacement, vectors, training, compute_force, compute_virials, compute_stress, compute_hessian, _28, )
    forces, virials, stress, hessian, edge_forces, = _29
    if compute_atomic_stresses:
      _31 = torch.__isnot__(edge_forces, None)
      _30 = _31
    else:
      _30 = False
    if _30:
      edge_forces1 = unchecked_cast(Tensor, edge_forces)
      _32 = _3(edge_forces1, data["edge_index"], vectors, (torch.size(positions))[0], data["batch"], cell, )
      atomic_virials0, atomic_stresses0, = _32
      edge_forces0, atomic_virials, atomic_stresses = edge_forces1, atomic_virials0, atomic_stresses0
    else:
      edge_forces0, atomic_virials, atomic_stresses = edge_forces, None, None
    _33 = annotate(Dict[str, Optional[Tensor]], {"energy": total_energy, "node_energy": node_energy, "interaction_energy": inter_e, "forces": forces, "edge_forces": edge_forces0, "virials": virials, "stress": stress, "atomic_virials": atomic_virials, "atomic_stresses": atomic_stresses, "hessian": hessian, "displacement": displacement, "node_feats": node_feats_out})
    return _33
