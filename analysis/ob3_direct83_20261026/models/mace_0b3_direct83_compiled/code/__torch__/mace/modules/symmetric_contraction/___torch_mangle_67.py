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
  contractions_weighting : __torch__.torch.nn.modules.container.___torch_mangle_65.ModuleList
  contractions_features : __torch__.torch.nn.modules.container.___torch_mangle_66.ModuleList
  weights : __torch__.torch.nn.modules.container.ParameterList
  graph_opt_main : __torch__.torch.fx.graph_module.___torch_mangle_64.GraphModule
  def forward(self: __torch__.mace.modules.symmetric_contraction.___torch_mangle_67.Contraction,
    x: Tensor,
    y: Tensor) -> Tensor:
    graph_opt_main = self.graph_opt_main
    correlation = self.correlation
    _0 = (self).U_tensors(correlation, )
    weights_max = self.weights_max
    out = (graph_opt_main).forward(_0, weights_max, x, y, )
    weights = self.weights
    _00 = getattr(weights, "0")
    _1 = getattr(weights, "1")
    contractions_weighting = self.contractions_weighting
    _01 = getattr(contractions_weighting, "0")
    _10 = getattr(contractions_weighting, "1")
    contractions_features = self.contractions_features
    _02 = getattr(contractions_features, "0")
    _11 = getattr(contractions_features, "1")
    correlation0 = self.correlation
    _2 = torch.sub(torch.sub(correlation0, 0), 1)
    c_tensor = (_01).forward((self).U_tensors(_2, ), _00, y, )
    c_tensor0 = torch.add(c_tensor, out)
    out0 = (_02).forward(c_tensor0, x, )
    correlation1 = self.correlation
    _3 = torch.sub(torch.sub(correlation1, 1), 1)
    c_tensor1 = (_10).forward((self).U_tensors(_3, ), _1, y, )
    c_tensor2 = torch.add(c_tensor1, out0)
    out1 = (_11).forward(c_tensor2, x, )
    _4 = torch.view(out1, [(torch.size(out1))[0], -1])
    return _4
  def U_tensors(self: __torch__.mace.modules.symmetric_contraction.___torch_mangle_67.Contraction,
    nu: int) -> Tensor:
    U_matrix_1 = self.U_matrix_1
    U_matrix_2 = self.U_matrix_2
    U_matrix_3 = self.U_matrix_3
    weights_0_zeroed = self.weights_0_zeroed
    weights_1_zeroed = self.weights_1_zeroed
    weights_max_zeroed = self.weights_max_zeroed
    _4 = annotate(Dict[str, Tensor], {})
    torch._set_item(_4, "U_matrix_1", U_matrix_1)
    torch._set_item(_4, "U_matrix_2", U_matrix_2)
    torch._set_item(_4, "U_matrix_3", U_matrix_3)
    torch._set_item(_4, "weights_0_zeroed", weights_0_zeroed)
    torch._set_item(_4, "weights_1_zeroed", weights_1_zeroed)
    torch._set_item(_4, "weights_max_zeroed", weights_max_zeroed)
    return _4[torch.format("U_matrix_{}", nu)]
