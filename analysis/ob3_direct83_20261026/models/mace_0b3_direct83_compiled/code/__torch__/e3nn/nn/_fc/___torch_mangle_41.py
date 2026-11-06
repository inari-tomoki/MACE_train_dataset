class FullyConnectedNet(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  hs : List[int]
  layer0 : __torch__.e3nn.nn._fc.___torch_mangle_34._Layer
  layer1 : __torch__.e3nn.nn._fc.___torch_mangle_37._Layer
  layer2 : __torch__.e3nn.nn._fc.___torch_mangle_40._Layer
  layer3 : __torch__.e3nn.nn._fc.___torch_mangle_27._Layer
  def forward(self: __torch__.e3nn.nn._fc.___torch_mangle_41.FullyConnectedNet,
    input: Tensor) -> Tensor:
    layer0 = self.layer0
    layer1 = self.layer1
    layer2 = self.layer2
    layer3 = self.layer3
    input0 = (layer0).forward(input, )
    input1 = (layer1).forward(input0, )
    input2 = (layer2).forward(input1, )
    return (layer3).forward(input2, )
  def __len__(self: __torch__.e3nn.nn._fc.___torch_mangle_41.FullyConnectedNet) -> int:
    return 4
