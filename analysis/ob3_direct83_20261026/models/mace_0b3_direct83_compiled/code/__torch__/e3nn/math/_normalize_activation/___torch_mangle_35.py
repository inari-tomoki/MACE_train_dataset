class normalize2mom(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.e3nn.math._normalize_activation.___torch_mangle_35.normalize2mom,
    x: Tensor) -> Tensor:
    _0 = torch.mul(torch.silu(x), CONSTANTS.c0)
    return _0
