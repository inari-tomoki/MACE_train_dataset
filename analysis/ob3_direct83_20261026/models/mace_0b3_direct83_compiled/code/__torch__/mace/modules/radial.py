class BesselBasis(Module):
  __parameters__ = ["bessel_weights", ]
  __buffers__ = ["r_max", "prefactor", ]
  bessel_weights : Tensor
  r_max : Tensor
  prefactor : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.mace.modules.radial.BesselBasis,
    x: Tensor) -> Tensor:
    bessel_weights = self.bessel_weights
    numerator = torch.sin(torch.mul(bessel_weights, x))
    prefactor = self.prefactor
    _0 = torch.mul(prefactor, torch.div(numerator, x))
    return _0
class AgnesiTransform(Module):
  __parameters__ = []
  __buffers__ = ["q", "p", "a", "covalent_radii", ]
  q : Tensor
  p : Tensor
  a : Tensor
  covalent_radii : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.mace.modules.radial.AgnesiTransform,
    x: Tensor,
    node_attrs: Tensor,
    edge_index: Tensor,
    atomic_numbers: Tensor) -> Tensor:
    sender = torch.select(edge_index, 0, 0)
    receiver = torch.select(edge_index, 0, 1)
    _1 = annotate(List[Optional[Tensor]], [torch.argmax(node_attrs, 1)])
    node_atomic_numbers = torch.unsqueeze(torch.index(atomic_numbers, _1), -1)
    _2 = annotate(List[Optional[Tensor]], [sender])
    _3 = torch.index(node_atomic_numbers, _2)
    Z_u = torch.to(_3, 4)
    _4 = annotate(List[Optional[Tensor]], [receiver])
    _5 = torch.index(node_atomic_numbers, _4)
    Z_v = torch.to(_5, 4)
    covalent_radii = self.covalent_radii
    _6 = annotate(List[Optional[Tensor]], [Z_u])
    _7 = torch.index(covalent_radii, _6)
    covalent_radii0 = self.covalent_radii
    _8 = annotate(List[Optional[Tensor]], [Z_v])
    _9 = torch.add(_7, torch.index(covalent_radii0, _8))
    r_0 = torch.mul(_9, 0.5)
    r_over_r_0 = torch.div(x, r_0)
    a = self.a
    q = self.q
    _10 = torch.mul(a, torch.pow(r_over_r_0, q))
    q0 = self.q
    p = self.p
    _11 = torch.pow(r_over_r_0, torch.sub(q0, p))
    _12 = torch.add(torch.div(_10, torch.add(_11, 1)), 1)
    return torch.reciprocal_(_12)
class PolynomialCutoff(Module):
  __parameters__ = []
  __buffers__ = ["p", "r_max", ]
  p : Tensor
  r_max : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  def forward(self: __torch__.mace.modules.radial.PolynomialCutoff,
    x: Tensor) -> Tensor:
    _13 = __torch__.mace.modules.radial.calculate_envelope
    r_max = self.r_max
    p = self.p
    return _13(x, r_max, torch.to(p, 3), )
class ZBLBasis(Module):
  __parameters__ = []
  __buffers__ = ["c", "p", "covalent_radii", "a_exp", "a_prefactor", ]
  c : Tensor
  p : Tensor
  covalent_radii : Tensor
  a_exp : Tensor
  a_prefactor : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  def forward(self: __torch__.mace.modules.radial.ZBLBasis,
    x: Tensor,
    node_attrs: Tensor,
    edge_index: Tensor,
    atomic_numbers: Tensor) -> Tensor:
    _14 = __torch__.mace.modules.radial.calculate_envelope
    _15 = __torch__.mace.tools.scatter.scatter_sum
    sender = torch.select(edge_index, 0, 0)
    receiver = torch.select(edge_index, 0, 1)
    _16 = annotate(List[Optional[Tensor]], [torch.argmax(node_attrs, 1)])
    node_atomic_numbers = torch.unsqueeze(torch.index(atomic_numbers, _16), -1)
    _17 = annotate(List[Optional[Tensor]], [sender])
    _18 = torch.index(node_atomic_numbers, _17)
    Z_u = torch.to(_18, 4)
    _19 = annotate(List[Optional[Tensor]], [receiver])
    _20 = torch.index(node_atomic_numbers, _19)
    Z_v = torch.to(_20, 4)
    a_prefactor = self.a_prefactor
    _21 = torch.mul(a_prefactor, 0.52900000000000003)
    a_exp = self.a_exp
    _22 = torch.pow(Z_u, a_exp)
    a_exp0 = self.a_exp
    _23 = torch.add(_22, torch.pow(Z_v, a_exp0))
    a = torch.div(_21, _23)
    r_over_a = torch.div(x, a)
    c = self.c
    _24 = torch.select(c, 0, 0)
    _25 = torch.mul(r_over_a, -3.2000000000000002)
    _26 = torch.mul(_24, torch.exp(_25))
    c0 = self.c
    _27 = torch.select(c0, 0, 1)
    _28 = torch.mul(r_over_a, -0.94230000000000003)
    _29 = torch.add(_26, torch.mul(_27, torch.exp(_28)))
    c1 = self.c
    _30 = torch.select(c1, 0, 2)
    _31 = torch.mul(r_over_a, -0.40279999999999999)
    _32 = torch.add(_29, torch.mul(_30, torch.exp(_31)))
    c2 = self.c
    _33 = torch.select(c2, 0, 3)
    _34 = torch.exp(torch.mul(r_over_a, -0.2016))
    phi = torch.add(_32, torch.mul(_33, _34))
    _35 = torch.mul(torch.mul(Z_u, 14.3996), Z_v)
    v_edges = torch.mul(torch.div(_35, x), phi)
    covalent_radii = self.covalent_radii
    _36 = annotate(List[Optional[Tensor]], [Z_u])
    _37 = torch.index(covalent_radii, _36)
    covalent_radii1 = self.covalent_radii
    _38 = annotate(List[Optional[Tensor]], [Z_v])
    r_max = torch.add(_37, torch.index(covalent_radii1, _38))
    p = self.p
    envelope = _14(x, r_max, p, )
    v_edges0 = torch.mul(torch.mul(v_edges, 0.5), envelope)
    V_ZBL = _15(v_edges0, receiver, 0, None, torch.size(node_attrs, 0), "sum", )
    return torch.squeeze(V_ZBL, -1)
def calculate_envelope(x: Tensor,
    r_max: Tensor,
    p: Tensor) -> Tensor:
  r_over_r_max = torch.div(x, r_max)
  _39 = torch.mul(torch.add(p, 1.), torch.add(p, 2.))
  _40 = torch.mul(torch.div(_39, 2.), torch.pow(r_over_r_max, p))
  _41 = torch.add(torch.neg(_40), 1.)
  _42 = torch.mul(p, torch.add(p, 2.))
  _43 = torch.pow(r_over_r_max, torch.add(p, 1))
  _44 = torch.add(_41, torch.mul(_42, _43))
  _45 = torch.div(torch.mul(p, torch.add(p, 1.)), 2)
  _46 = torch.pow(r_over_r_max, torch.add(p, 2))
  envelope = torch.sub(_44, torch.mul(_45, _46))
  _47 = torch.mul(envelope, torch.lt(x, r_max))
  return _47
