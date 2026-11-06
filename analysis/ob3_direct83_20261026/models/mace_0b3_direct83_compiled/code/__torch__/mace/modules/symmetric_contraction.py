class SymmetricContraction(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  irreps_in : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  irreps_out : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  internal_weights : bool
  shared_weights : NoneType
  contractions : __torch__.torch.nn.modules.container.___torch_mangle_58.ModuleList
  def forward(self: __torch__.mace.modules.symmetric_contraction.SymmetricContraction,
    x: Tensor,
    y: Tensor) -> Tensor:
    outs = annotate(List[Tensor], [])
    contractions = self.contractions
    _0 = getattr(contractions, "0")
    _1 = getattr(contractions, "1")
    _2 = torch.append(outs, (_0).forward(x, y, ))
    _3 = torch.append(outs, (_1).forward(x, y, ))
    return torch.cat(outs, -1)
class Contraction(Module):
  __parameters__ = ["weights_max", ]
  __buffers__ = ["U_matrix_1", "U_matrix_2", "U_matrix_3", "weights_0_zeroed", "weights_1_zeroed", "weights_max_zeroed", ]
  weights_max : Tensor
  U_matrix_1 : Tensor
  U_matrix_2 : Tensor
  U_matrix_3 : Tensor
  weights_0_zeroed : Tensor
  weights_1_zeroed : Tensor
  weights_max_zeroed : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  num_features : int
  coupling_irreps : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  correlation : int
  contractions_weighting : __torch__.torch.nn.modules.container.ModuleList
  contractions_features : __torch__.torch.nn.modules.container.___torch_mangle_49.ModuleList
  weights : __torch__.torch.nn.modules.container.ParameterList
  graph_opt_main : __torch__.torch.fx.graph_module.___torch_mangle_48.GraphModule
  def forward(self: __torch__.mace.modules.symmetric_contraction.Contraction,
    x: Tensor,
    y: Tensor) -> Tensor:
    graph_opt_main = self.graph_opt_main
    correlation = self.correlation
    _2 = (self).U_tensors(correlation, )
    weights_max = self.weights_max
    out = (graph_opt_main).forward(_2, weights_max, x, y, )
    weights = self.weights
    _0 = getattr(weights, "0")
    _1 = getattr(weights, "1")
    contractions_weighting = self.contractions_weighting
    _04 = getattr(contractions_weighting, "0")
    _13 = getattr(contractions_weighting, "1")
    contractions_features = self.contractions_features
    _05 = getattr(contractions_features, "0")
    _14 = getattr(contractions_features, "1")
    correlation0 = self.correlation
    _3 = torch.sub(torch.sub(correlation0, 0), 1)
    c_tensor = (_04).forward((self).U_tensors(_3, ), _0, y, )
    c_tensor0 = torch.add(c_tensor, out)
    out0 = (_05).forward(c_tensor0, x, )
    correlation1 = self.correlation
    _4 = torch.sub(torch.sub(correlation1, 1), 1)
    c_tensor1 = (_13).forward((self).U_tensors(_4, ), _1, y, )
    c_tensor2 = torch.add(c_tensor1, out0)
    out1 = (_14).forward(c_tensor2, x, )
    _5 = torch.view(out1, [(torch.size(out1))[0], -1])
    return _5
  def U_tensors(self: __torch__.mace.modules.symmetric_contraction.Contraction,
    nu: int) -> Tensor:
    U_matrix_1 = self.U_matrix_1
    U_matrix_2 = self.U_matrix_2
    U_matrix_3 = self.U_matrix_3
    weights_0_zeroed = self.weights_0_zeroed
    weights_1_zeroed = self.weights_1_zeroed
    weights_max_zeroed = self.weights_max_zeroed
    _6 = annotate(Dict[str, Tensor], {})
    torch._set_item(_6, "U_matrix_1", U_matrix_1)
    torch._set_item(_6, "U_matrix_2", U_matrix_2)
    torch._set_item(_6, "U_matrix_3", U_matrix_3)
    torch._set_item(_6, "weights_0_zeroed", weights_0_zeroed)
    torch._set_item(_6, "weights_1_zeroed", weights_1_zeroed)
    torch._set_item(_6, "weights_max_zeroed", weights_max_zeroed)
    return _6[torch.format("U_matrix_{}", nu)]
