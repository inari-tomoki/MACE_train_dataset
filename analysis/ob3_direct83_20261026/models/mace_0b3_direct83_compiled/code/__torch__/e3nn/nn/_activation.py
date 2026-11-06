class Activation(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  acts : __torch__.torch.nn.modules.container.___torch_mangle_76.ModuleList
  def forward(self: __torch__.e3nn.nn._activation.Activation,
    features: Tensor) -> Tensor:
    acts = self.acts
    _0 = getattr(acts, "0")
    _1 = (_0).forward(torch.narrow(features, -1, 0, 16), )
    return _1
