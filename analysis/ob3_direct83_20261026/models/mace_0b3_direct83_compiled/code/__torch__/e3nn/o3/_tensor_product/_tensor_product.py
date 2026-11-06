class TensorProduct(Module):
  __parameters__ = []
  __buffers__ = ["weight", "output_mask", ]
  weight : Tensor
  output_mask : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  irreps_in1 : Tuple[Tuple[int, Tuple[int, int]]]
  irreps_in2 : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  irreps_out : Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
  instructions : List[Any]
  _in1_dim : int
  _in2_dim : int
  internal_weights : bool
  shared_weights : bool
  _specialized_code : bool
  _optimize_einsums : bool
  weight_numel : int
  _profiling_str : str
  _compiled_main_left_right : __torch__.torch.fx.graph_module.___torch_mangle_4.GraphModule
  _compiled_main_right : __torch__.torch.fx.graph_module.tp_forward
  def forward(self: __torch__.e3nn.o3._tensor_product._tensor_product.TensorProduct,
    x: Tensor,
    y: Tensor,
    weight: Optional[Tensor]=None) -> Tensor:
    _0 = "AssertionError: Incorrect last dimension for x"
    _1 = "AssertionError: Incorrect last dimension for y"
    _2 = (torch.size(x))[-1]
    _in1_dim = self._in1_dim
    if torch.eq(_2, _in1_dim):
      pass
    else:
      ops.prim.RaiseException(_0)
    _3 = (torch.size(y))[-1]
    _in2_dim = self._in2_dim
    if torch.eq(_3, _in2_dim):
      pass
    else:
      ops.prim.RaiseException(_1)
    real_weight = (self)._get_weights(weight, )
    _compiled_main_left_right = self._compiled_main_left_right
    _4 = (_compiled_main_left_right).forward(x, y, real_weight, )
    return _4
  def right(self: __torch__.e3nn.o3._tensor_product._tensor_product.TensorProduct,
    y: Tensor,
    weight: Optional[Tensor]=None) -> NoneType:
    _5 = "AssertionError: Incorrect last dimension for y"
    _6 = (torch.size(y))[-1]
    _in2_dim = self._in2_dim
    if torch.eq(_6, _in2_dim):
      pass
    else:
      ops.prim.RaiseException(_5)
    real_weight = (self)._get_weights(weight, )
    _compiled_main_right = self._compiled_main_right
    _7 = (_compiled_main_right).forward(y, real_weight, )
    return _7
  def _get_weights(self: __torch__.e3nn.o3._tensor_product._tensor_product.TensorProduct,
    weight: Optional[Tensor]) -> Tensor:
    _8 = "Weights must be provided when the TensorProduct does not have `internal_weights`"
    _9 = "AssertionError: Invalid weight shape"
    _10 = "AssertionError: When shared weights is false, weights must have batch dimension"
    if torch.__is__(weight, None):
      weight_numel = self.weight_numel
      if torch.gt(weight_numel, 0):
        internal_weights = self.internal_weights
        _12 = torch.__not__(internal_weights)
      else:
        _12 = False
      if _12:
        ops.prim.RaiseException(_8, "builtins.RuntimeError")
      else:
        pass
      _11 = self.weight
    else:
      weight0 = unchecked_cast(Tensor, weight)
      shared_weights = self.shared_weights
      if shared_weights:
        _13 = torch.size(weight0)
        weight_numel0 = self.weight_numel
        if torch.eq(_13, [weight_numel0]):
          pass
        else:
          ops.prim.RaiseException(_9)
      else:
        _14 = (torch.size(weight0))[-1]
        weight_numel1 = self.weight_numel
        if torch.eq(_14, weight_numel1):
          pass
        else:
          ops.prim.RaiseException(_9)
        if torch.gt(torch.dim(weight0), 1):
          pass
        else:
          ops.prim.RaiseException(_10)
      _11 = weight0
    return _11
