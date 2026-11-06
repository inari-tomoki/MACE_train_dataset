class _Layer(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  act : NoneType
  h_in : float
  h_out : float
  var_in : float
  var_out : float
  _profiling_str : str
  def forward(self: __torch__.e3nn.nn._fc.___torch_mangle_27._Layer,
    x: Tensor) -> Tensor:
    weight = self.weight
    h_in = self.h_in
    var_in = self.var_in
    _0 = torch.mul(h_in, var_in)
    var_out = self.var_out
    _1 = torch.pow(torch.div(_0, var_out), 0.5)
    w = torch.div(weight, _1)
    return torch.matmul(x, w)
