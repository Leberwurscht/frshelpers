import functools

from . import ops

globals().update({funcname: functools.partial(getattr(ops,funcname), use_jax=True) for funcname in ops.__all__})

__all__ = ops.__all__
