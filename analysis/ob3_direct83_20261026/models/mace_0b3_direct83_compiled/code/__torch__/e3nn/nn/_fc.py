class FullyConnectedNet(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  hs : List[int]
  layer0 : __torch__.e3nn.nn._fc._Layer
  layer1 : __torch__.e3nn.nn._fc.___torch_mangle_23._Layer
  layer2 : __torch__.e3nn.nn._fc.___torch_mangle_26._Layer
  layer3 : __torch__.e3nn.nn._fc.___torch_mangle_27._Layer
  def forward(self: __torch__.e3nn.nn._fc.FullyConnectedNet,
    input: Tensor) -> Tensor:
    layer0 = self.layer0
    layer1 = self.layer1
    layer2 = self.layer2
    layer3 = self.layer3
    input0 = (layer0).forward(input, )
    input1 = (layer1).forward(input0, )
    input2 = (layer2).forward(input1, )
    return (layer3).forward(input2, )
  def __len__(self: __torch__.e3nn.nn._fc.FullyConnectedNet) -> int:
    return 4
class _Layer(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  h_in : float
  h_out : float
  var_in : float
  var_out : float
  _profiling_str : str
  act : __torch__.e3nn.math._normalize_activation.normalize2mom
  def forward(self: __torch__.e3nn.nn._fc._Layer,
    x: Tensor) -> Tensor:
    weight = self.weight
    h_in = self.h_in
    var_in = self.var_in
    _0 = torch.pow(torch.mul(h_in, var_in), 0.5)
    w = torch.div(weight, _0)
    x0 = torch.matmul(x, w)
    act = self.act
    x1 = (act).forward(x0, )
    var_out = self.var_out
    x2 = torch.mul(x1, torch.pow(var_out, 0.5))
    return x2
