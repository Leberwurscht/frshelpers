"""
  These functions return callbacks compatible with chunkiter.apply.
  They are written in a way so that they are compatible with jax.
"""

import fourioso
import numpy

try:
  import fourioso.jax
  import jax.numpy
except ModuleNotFoundError:
  pass

def subtract_mean(use_jax=False):
  return lambda chunk: chunk - chunk.mean(axis=-1)[:,None]

def multiply(window, use_jax=False):
  return lambda chunk: chunk*window[None,:]

def fourier_transform(axis, use_jax=False):
  if use_jax: transform = fourioso.jax.transform
  else: transform = fourioso.transform

  return lambda chunk: transform(axis[None,:]-axis[axis.size//2],chunk,return_axis=False)

def inverse_fourier_transform(axis, use_jax=False):
  if use_jax: itransform = fourioso.jax.itransform
  else: itransform = fourioso.itransform

  return lambda chunk: itransform(axis[None,:]-axis[axis.size//2],chunk,return_axis=False)

def complex_to_polar(use_jax=False):
  if use_jax: np = jax.numpy
  else: np = numpy

  return lambda chunk: (abs(chunk), np.nan_to_num(np.angle(chunk)))

def polar_to_complex(use_jax=False):
  if use_jax: np = jax.numpy
  else: np = numpy

  return lambda chunk: chunk[0]*np.exp(1j*chunk[1])

def unwrap(N_pts, use_jax=False):
  if use_jax: np = jax.numpy
  else: np = numpy

  initial_angle = np.full(N_pts, np.nan)

  def process_chunk(chunk, carry):
    angles, initial_angle = chunk, carry

    initial_angle = np.where(np.isfinite(initial_angle), initial_angle, angles[0,:])

    angles = np.concatenate((initial_angle[None,:], chunk), axis=0)
    angles = np.unwrap(angles, axis=0)

    chunk, carry = angles[1:,:], angles[-1,:]
    return chunk, carry

  process_chunk.has_carry = True
  process_chunk.initial_carry = initial_angle

  return process_chunk

def cancel_polyfit(nu, weights, degree=1, use_jax=False, operation="subtract", initial=None, asymmetric=False):
  if use_jax: np = jax.numpy
  else: np = numpy

  if operation not in ["subtract","divide"]: raise ValueError("operation must be 'subtract' or 'divide'")
  operation = (lambda x,y: (x-y)) if operation=="subtract" else (lambda x,y: np.nan_to_num(x/y))

  if initial is None: initial = np.full(weights.size, np.nan)

  def process_chunk(chunk, carry):
    if degree is None: return chunk, carry

    initial = carry
    initial = np.where(np.isfinite(carry), carry, chunk[0,:])
    carry = initial

    fit = np.polyfit(nu, operation(chunk, initial[None,:]).T, degree, w=weights)

    fitted = 0
    for i in range(degree+1):
      fitted = fitted + fit[i,:][:,None]*abs(nu[None,:])**(degree-i)
    
    if asymmetric: fitted = fitted * np.sign(nu[None,:])

    new_chunk = operation(chunk, fitted)

    return new_chunk, carry

  process_chunk.has_carry = True
  process_chunk.initial_carry = initial

  return process_chunk

def identity(use_jax=False):
  return lambda x: x

__all__ = ["subtract_mean", "multiply", "fourier_transform", "inverse_fourier_transform", "complex_to_polar", "polar_to_complex", "unwrap", "cancel_polyfit", "identity"]
