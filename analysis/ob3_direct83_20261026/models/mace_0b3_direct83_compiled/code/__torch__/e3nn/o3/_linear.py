class Linear(Module):
  __parameters__ = ["weight", ]
  __buffers__ = ["bias", "output_mask", ]
  weight : Tensor
  bias : Tensor
  output_mask : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  internal_weights : bool
  shared_weights : bool
  irreps_in : Tuple[Tuple[int, Tuple[int, int]]]
  irreps_out : Tuple[Tuple[int, Tuple[int, int]]]
  instructions : List[Tuple[int, int, Tuple[int, int], float]]
  _optimize_einsums : bool
  weight_numel : int
  bias_numel : int
  _compiled_main : __torch__.torch.fx.graph_module.GraphModule
  def forward(self: __torch__.e3nn.o3._linear.Linear,
    features: Tensor,
    weight: Optional[Tensor]=None,
    bias: Optional[Tensor]=None) -> Tensor:
    _0 = "Weights must be provided when internal_weights = False"
    _1 = "Biases must be provided when internal_weights = False"
    if torch.__is__(weight, None):
      weight_numel = self.weight_numel
      if torch.gt(weight_numel, 0):
        internal_weights = self.internal_weights
        _2 = torch.__not__(internal_weights)
      else:
        _2 = False
      if _2:
        ops.prim.RaiseException(_0, "builtins.RuntimeError")
      else:
        pass
      weight0 = self.weight
    else:
      weight0 = unchecked_cast(Tensor, weight)
    if torch.__is__(bias, None):
      bias_numel = self.bias_numel
      if torch.gt(bias_numel, 0):
        internal_weights0 = self.internal_weights
        _4 = torch.__not__(internal_weights0)
        _3 = _4
      else:
        _3 = False
      if _3:
        ops.prim.RaiseException(_1, "builtins.RuntimeError")
      else:
        pass
      bias0 = self.bias
    else:
      bias0 = unchecked_cast(Tensor, bias)
    _compiled_main = self._compiled_main
    _5 = (_compiled_main).forward(features, weight0, bias0, )
    return _5
