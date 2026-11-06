def prepare_graph(data: Dict[str, Tensor],
    compute_virials: bool=False,
    compute_stress: bool=False,
    compute_displacement: bool=False,
    lammps_mliap: bool=False) -> __torch__.mace.modules.utils.GraphContext:
  _0 = __torch__.mace.modules.utils.get_symmetric_displacement
  _1 = __torch__.mace.modules.utils.get_edge_vectors_and_lengths
  if torch.__contains__(data, "head"):
    _2 = data["head"]
    _3 = annotate(List[Optional[Tensor]], [data["batch"]])
    node_heads = torch.index(_2, _3)
  else:
    node_heads = torch.zeros_like(data["batch"])
  _4 = __torch__.torch.compiler.is_compiling()
  if torch.__not__(_4):
    _5 = torch.requires_grad_(data["positions"])
  else:
    pass
  positions = data["positions"]
  cell = data["cell"]
  num_atoms_arange = torch.arange((torch.size(positions))[0], dtype=None, layout=None, device=ops.prim.device(positions))
  num_graphs = torch.sub(torch.numel(data["ptr"]), 1)
  _6 = ops.prim.dtype(positions)
  _7 = ops.prim.device(positions)
  displacement = torch.zeros([num_graphs, 3, 3], dtype=_6, layout=None, device=_7)
  if compute_virials:
    _8 = True
  else:
    _8 = compute_stress
  if _8:
    _9 = True
  else:
    _9 = compute_displacement
  if _9:
    _10 = _0(positions, data["unit_shifts"], cell, data["edge_index"], num_graphs, data["batch"], torch.get(data, "displacement"), )
    p, s, displacement1, = _10
    torch._set_item(data, "positions", p)
    torch._set_item(data, "shifts", s)
    displacement0 = displacement1
  else:
    displacement0 = displacement
  _11 = _1(data["positions"], data["edge_index"], data["shifts"], False, 1.0000000000000001e-09, )
  vectors, lengths, = _11
  ikw = __torch__.mace.modules.utils.InteractionKwargs(None, (0, 0))
  _12 = __torch__.mace.modules.utils.GraphContext(False, num_graphs, num_atoms_arange, displacement0, positions, vectors, lengths, cell, node_heads, ikw)
  return _12
def get_outputs(energy: Tensor,
    positions: Tensor,
    cell: Tensor,
    displacement: Optional[Tensor],
    vectors: Optional[Tensor]=None,
    training: bool=False,
    compute_force: bool=True,
    compute_virials: bool=True,
    compute_stress: bool=True,
    compute_hessian: bool=False,
    compute_edge_forces: bool=False) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
  _13 = __torch__.mace.modules.utils.compute_forces_virials
  _14 = __torch__.mace.modules.utils.compute_forces
  _15 = "AssertionError: Forces must be computed to get the hessian"
  _16 = "This Python function is annotated to be ignored and cannot be run"
  _17 = uninitialized(Tensor)
  if compute_virials:
    _18 = True
  else:
    _18 = compute_stress
  if _18:
    _19 = torch.__isnot__(displacement, None)
  else:
    _19 = False
  if _19:
    displacement2 = unchecked_cast(Tensor, displacement)
    if training:
      _20 = True
    else:
      _20 = compute_hessian
    if _20:
      _21 = True
    else:
      _21 = compute_edge_forces
    _22 = _13(energy, positions, displacement2, cell, _21, compute_stress, )
    forces0, virials0, stress0, = _22
    forces, virials, stress = forces0, virials0, stress0
  else:
    if compute_force:
      if training:
        _23 = True
      else:
        _23 = compute_hessian
      if _23:
        _24 = True
      else:
        _24 = compute_edge_forces
      forces1 : Optional[Tensor] = _14(energy, positions, _24, )
    else:
      forces1 = None
    forces, virials, stress = forces1, None, None
  if compute_hessian:
    if torch.__isnot__(forces, None):
      forces3 = unchecked_cast(Tensor, forces)
    else:
      ops.prim.RaiseException(_15)
      forces3 = _17
    ops.prim.RaiseException(_16)
    forces2, hessian = forces3, _17
  else:
    forces2, hessian = forces, None
  if compute_edge_forces:
    _25 = torch.__isnot__(vectors, None)
  else:
    _25 = False
  if _25:
    vectors0 = unchecked_cast(Tensor, vectors)
    if training:
      _26 = True
    else:
      _26 = compute_hessian
    edge_forces0 = _14(energy, vectors0, _26, )
    edge_forces : Optional[Tensor] = torch.mul(edge_forces0, -1)
  else:
    edge_forces = None
  _27 = (forces2, virials, stress, hessian, edge_forces)
  return _27
def get_atomic_virials_stresses(edge_forces: Tensor,
    edge_index: Tensor,
    vectors: Tensor,
    num_atoms: int,
    batch: Tensor,
    cell: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
  _28 = __torch__.mace.tools.scatter.scatter_sum
  edge_virial = torch.einsum("zi,zj->zij", [edge_forces, vectors])
  atom_virial_sender = _28(edge_virial, torch.select(edge_index, 0, 0), 0, None, num_atoms, "sum", )
  atom_virial_receiver = _28(edge_virial, torch.select(edge_index, 0, 1), 0, None, num_atoms, "sum", )
  _29 = torch.add(atom_virial_sender, atom_virial_receiver)
  atom_virial = torch.div(_29, 2)
  _30 = torch.add(atom_virial, torch.transpose(atom_virial, -1, -2))
  atom_virial0 = torch.div(_30, 2)
  cell0 = torch.view(cell, [-1, 3, 3])
  volume = torch.unsqueeze(torch.abs(torch.linalg_det(cell0)), -1)
  _31 = annotate(List[Optional[Tensor]], [batch])
  atom_volume = torch.view(torch.index(volume, _31), [-1, 1, 1])
  atom_stress = torch.div(atom_virial0, atom_volume)
  _32 = torch.lt(torch.abs(atom_stress), 10000000000)
  atom_stress0 = torch.where(_32, atom_stress, torch.zeros_like(atom_stress))
  _33 = (torch.mul(atom_virial0, -1), atom_stress0)
  return _33
class GraphContext(NamedTuple):
  is_lammps : bool
  num_graphs : int
  num_atoms_arange : Tensor
  displacement : Optional[Tensor]
  positions : Tensor
  vectors : Tensor
  lengths : Tensor
  cell : Tensor
  node_heads : Tensor
  interaction_kwargs : __torch__.mace.modules.utils.InteractionKwargs
class InteractionKwargs(NamedTuple):
  lammps_class : Optional[Tensor]
  lammps_natoms : Tuple[int, int]
def get_symmetric_displacement(positions: Tensor,
    unit_shifts: Tensor,
    cell: Optional[Tensor],
    edge_index: Tensor,
    num_graphs: int,
    batch: Tensor,
    displacement: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
  if torch.__is__(cell, None):
    _34 = torch.mul(num_graphs, 3)
    _35 = ops.prim.dtype(positions)
    _36 = ops.prim.device(positions)
    cell2 = torch.zeros([_34, 3], dtype=_35, layout=None, device=_36)
    cell1 = cell2
  else:
    cell1 = unchecked_cast(Tensor, cell)
  sender = torch.select(edge_index, 0, 0)
  if torch.__is__(displacement, None):
    _37 = ops.prim.dtype(positions)
    _38 = ops.prim.device(positions)
    displacement4 = torch.zeros([num_graphs, 3, 3], dtype=_37, layout=None, device=_38)
    displacement5 = torch.add(displacement4, torch.mul(torch.sum(positions), 0.))
    displacement3 = displacement5
  else:
    displacement6 = unchecked_cast(Tensor, displacement)
    displacement3 = displacement6
  _39 = torch.transpose(displacement3, -1, -2)
  symmetric_displacement = torch.mul(torch.add(displacement3, _39), 0.5)
  _40 = annotate(List[Optional[Tensor]], [batch])
  _41 = torch.index(symmetric_displacement, _40)
  _42 = torch.einsum("be,bec->bc", [positions, _41])
  positions0 = torch.add(positions, _42)
  cell3 = torch.view(cell1, [-1, 3, 3])
  _43 = torch.matmul(cell3, symmetric_displacement)
  cell4 = torch.add(cell3, _43)
  _44 = annotate(List[Optional[Tensor]], [sender])
  _45 = annotate(List[Optional[Tensor]], [torch.index(batch, _44)])
  _46 = [unit_shifts, torch.index(cell4, _45)]
  shifts = torch.einsum("be,bec->bc", _46)
  return (positions0, shifts, displacement3)
def get_edge_vectors_and_lengths(positions: Tensor,
    edge_index: Tensor,
    shifts: Tensor,
    normalize: bool=False,
    eps: float=1.0000000000000001e-09) -> Tuple[Tensor, Tensor]:
  sender = torch.select(edge_index, 0, 0)
  receiver = torch.select(edge_index, 0, 1)
  _47 = annotate(List[Optional[Tensor]], [receiver])
  _48 = torch.index(positions, _47)
  _49 = annotate(List[Optional[Tensor]], [sender])
  _50 = torch.sub(_48, torch.index(positions, _49))
  vectors = torch.add(_50, shifts)
  lengths = torch.linalg_norm(vectors, None, [-1], True)
  if normalize:
    vectors_normed = torch.div(vectors, torch.add(lengths, eps))
    _51 = (vectors_normed, lengths)
  else:
    _51 = (vectors, lengths)
  return _51
def compute_forces_virials(energy: Tensor,
    positions: Tensor,
    displacement: Tensor,
    cell: Tensor,
    training: bool=True,
    compute_stress: bool=False) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
  grad_outputs = annotate(List[Optional[Tensor]], [torch.ones_like(energy)])
  _52 = torch.autograd.grad([energy], [positions, displacement], grad_outputs, training, training, True)
  forces, virials, = _52
  stress = torch.zeros_like(displacement)
  if compute_stress:
    _53 = torch.__isnot__(virials, None)
  else:
    _53 = False
  if _53:
    virials2 = unchecked_cast(Tensor, virials)
    cell5 = torch.view(cell, [-1, 3, 3])
    volume = torch.unsqueeze(torch.abs(torch.linalg_det(cell5)), -1)
    stress2 = torch.div(virials2, torch.view(volume, [-1, 1, 1]))
    _54 = torch.lt(torch.abs(stress2), 10000000000)
    stress3 = torch.where(_54, stress2, torch.zeros_like(stress2))
    virials1, stress1 = virials2, stress3
  else:
    virials1, stress1 = virials, stress
  if torch.__is__(forces, None):
    forces4 = torch.zeros_like(positions)
  else:
    forces4 = unchecked_cast(Tensor, forces)
  if torch.__is__(virials1, None):
    virials3 = torch.zeros([1, 3, 3])
  else:
    virials3 = unchecked_cast(Tensor, virials1)
  _55 = (torch.mul(forces4, -1), torch.mul(virials3, -1), stress1)
  return _55
def compute_forces(energy: Tensor,
    positions: Tensor,
    training: bool=True) -> Tensor:
  grad_outputs = annotate(List[Optional[Tensor]], [torch.ones_like(energy)])
  _56 = torch.autograd.grad([energy], [positions], grad_outputs, training, training, True)
  gradient = _56[0]
  if torch.__is__(gradient, None):
    _57 = torch.zeros_like(positions)
  else:
    gradient0 = unchecked_cast(Tensor, gradient)
    _57 = torch.mul(gradient0, -1)
  return _57
