class FullyConnectedNet(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  hs : List[int]
  layer0 : __torch__.e3nn.nn._fc.___torch_mangle_27._Layer
  def forward(self: __torch__.e3nn.nn._fc.___torch_mangle_29.FullyConnectedNet,
    input: Tensor) -> Tensor:
    layer0 = self.layer0
    return (layer0).forward(input, )
  def __len__(self: __torch__.e3nn.nn._fc.___torch_mangle_29.FullyConnectedNet) -> int:
    return 1
