#!/usr/bin/env python

import os, sys, functools, itertools

import numpy as np
import matplotlib.pyplot as plt

import chunkiter, chunkiter.tools # pip install git+https://gitlab.com/leberwurscht/chunkiter.git@v0.0.19
import fourioso, fourioso.tools # pip install git+https://gitlab.com/leberwurscht/fourioso.git@v0.0.5
import binreduce # pip install git+https://gitlab.com/leberwurscht/binreduce.git@v0.0.7
import frshelpers.plot # pip install git+https://gitlab.com/leberwurscht/frshelpers.git@v0.0.2

### initialize JAX if available (for multi-core CPU / GPU acceleration)
try:
  import jaxl
  from frshelpers import jaxops as ops
  jit = jax.jit
except ModuleNotFoundError:
  from frshelpers import ops
  jit = lambda func: func

### settings

filename = "/tmp/SampleFW-interpolated.h5"

average_traces = 8
nu_center = 34.5e12
nu_span = 2*5e12
bin_centers = np.linspace(29e12, 39e12, 12)

nu_center_normalization = 30e12
nu_span_normalization = 20e12

unit_firsttraces = 50e-2
unit = 5e-2

ylim = 1e-5, 5e-2

### read in the data

def read_csv_from_folder(folder):
  """
    python generator function yielding from files named 0.csv, 1.csv, ... from a folder
  """

  for i in itertools.count():
    try:
      data = np.loadtxt(os.path.join(folder, "{}.csv".format(i)))
    except FileNotFoundError:
      break

    yield data[None,:] # reshape to 1 x ... to respect chunkiter convention

if filename.endswith(".h5"): # ... from an HDF5 file with already Fourier-transformed traces
  spectra = chunkiter.IterableH5Chunks(filename, "data")
  nu = chunkiter.array_from_h5(filename, "nu") # HDF5 file must also contain the frequency axis (unit: Hertz)

else: # ... from a folder containing the traces as individual CSV files named 0.csv, 1.csv, ...
  delay = np.loadtxt(os.path.join(folder, "delay.csv")) # folder must also contain delay axis as delay.csv (unit: seconds)

  traces = read_csv_from_folder(filename)
  traces = chunkiter.rechunk(traces, 1024)

  window = fourioso.tools.piecewise_cossqr(np.linspace(0,1,delay.size), [0,0.05,0.95,1], [0, 1, 1, 0])
  operations = jit(chunkiter.chain(
    ops.subtract_mean(),
    ops.multiply(window),
    ops.fourier_transform(axis=delay),
  ))
  spectra = chunkiter.apply(operations, traces)

  nu = fourioso.transform(delay)

### checks

# make sure frequency axis is evenly spaced (needed for Fourier transform)
assert np.allclose(np.diff(nu)/np.diff(nu).mean(), 1, rtol=0, atol=1e-5)

# make sure we don't have NaN values
def check(arr):
  assert np.all(np.isfinite(arr)), "invalid value (e.g., NaN or inf) in input data"
  return arr
spectra = (check(i) for i in spectra)

### operations on data
weights = abs(nu-nu_center)<nu_span/2

# first step: do amplitude normalization and spectral phase normalization on each trace
operations = jit(chunkiter.chain(
  ops.complex_to_polar(),
  chunkiter.per_entry(ops.identity(), ops.unwrap(nu.size)),
  chunkiter.per_entry(ops.cancel_polyfit(nu, weights=weights, degree=0, operation="divide"), ops.cancel_polyfit(nu, weights=weights, degree=1, operation="subtract", asymmetric=True)),
))
spectra = chunkiter.apply(operations, spectra)

# cache result
#spectra = chunkiter.cache(((np.array(amplitude_part),np.array(phase_part)) for amplitude_part,phase_part in spectra))

# save amplitudes and phases before averaging for later
(first_amps, first_phases), spectra = chunkiter.tools.peek(iter(spectra))

# second step: average several consecutive traces to bring noise down, and do the normalizations again
spectra = chunkiter.apply(ops.polar_to_complex(), spectra)
spectra = chunkiter.tools.batchavg(spectra, average_traces, allow_remainder=True)

operations = jit(chunkiter.chain(
  ops.complex_to_polar(),
  chunkiter.per_entry(ops.identity(), ops.unwrap(nu.size)),
  chunkiter.per_entry(ops.cancel_polyfit(nu, weights=weights, degree=0, operation="divide"), ops.cancel_polyfit(nu, weights=weights, degree=1, operation="subtract", asymmetric=True)),
))
spectra = chunkiter.apply(operations, spectra)

# time domain plot
if 0:
  # cache result so that we can apply a different set of operations without consuming the iterator
  spectra = chunkiter.cache(((np.array(amplitude_part),np.array(phase_part)) for amplitude_part,phase_part in spectra))

  avg = 8000 # set this to the minimum of the Allan deviation

  delay = fourioso.itransform(nu-nu[nu.size//2])
  #nuwindow = fourioso.tools.piecewise_cossqr(np.linspace(nu.min(),nu.max(),nu.size), [27e12, 28e12, 39e12, 40e12], [0, 1, 1, 0]) # adapt
  operations = (chunkiter.chain(
    #ops.multiply(nuwindow), # can be tried out to improve results
    ops.polar_to_complex(),
    ops.inverse_fourier_transform(axis=nu),
  ))
  traces_td = chunkiter.apply(operations, spectra)
  traces_td = chunkiter.tools.batchavg(traces_td, max(1,int(avg/average_traces)), allow_remainder=True)
  first_traces, traces_td = chunkiter.tools.peek(iter(traces_td))
  plt.figure()
  maxvalue = abs(first_traces[0,:]).max()
  t0 = delay[np.argmax(abs(first_traces[0,:]))]
  plt.plot(delay-t0, abs(first_traces[0,:])/maxvalue, label="first")
  plt.plot(delay-t0, abs(first_traces[0,:]-first_traces[1,:])/maxvalue, label="difference")
  plt.plot(delay-t0, abs(abs(first_traces[0,:])-abs(first_traces[1,:]))/maxvalue, label="difference abs")
  plt.legend()
  plt.yscale("log")
  plt.ylim((1e-7,1))
  plt.title("avg = {}".format(avg))
  plt.grid()
  plt.savefig("timedomain_dynamicrange.png")

# compute relative amplitude change and phase for a few frequency bins
bincenters_i = np.searchsorted(nu, bin_centers)
amplitudes, phases = chunkiter.tools.concatenate((amp[:,bincenters_i],phase[:,bincenters_i]) for amp,phase in spectra)

phases_beforeavg = first_phases[:,bincenters_i]
phases_beforeavg = phases_beforeavg - phases_beforeavg[[0],:]
phases = phases - phases.mean(axis=0)[None,:]

relampchanges_beforeavg = first_amps[:,bincenters_i]
relampchanges_beforeavg = np.nan_to_num( (relampchanges_beforeavg-relampchanges_beforeavg[[0],:]) / relampchanges_beforeavg[[0],:] )
relampchanges = np.nan_to_num( (amplitudes-amplitudes[[0],:]) / amplitudes[[0],:] )

# plotting
fig = plt.figure(figsize=(16,9), constrained_layout=True)
gs = fig.add_gridspec(4,2)

for column, (data_beforeavg,data) in enumerate(zip([relampchanges_beforeavg, phases_beforeavg], [relampchanges, phases])):
  ax = fig.add_subplot(gs[0,column])
  where = abs(nu-nu_center_normalization)<nu_span_normalization/2

  ax.set_xlim((27,41))
  ax.set_xlabel("$\\nu$ (THz)")
  ax.set_title("first trace")

  if column==0:
    ax.plot(nu[where]/1e12, first_amps[0,:][where]**2, 'k')
    ax.set_ylabel("PSD (arb. u.)")
  else:
    phase = frshelpers.plot.spectral_phase_without_CEP_and_GD(nu, first_amps[0,:]*np.exp(1j*first_phases[0,:]), where)
    ax.plot(nu[where]/1e12, phase[where], 'k')
    ax.set_ylabel("phase (rad)")

    axt = ax.twinx()
    axt.plot(nu[where]/1e12, first_amps[0,:][where]**2, 'k:')
    axt.tick_params(labelright=False, right=False)

  ax = fig.add_subplot(gs[1,column], sharex=ax)
  corridor_label = "{:.1f}%".format(unit_firsttraces/1e-2) if column==0 else "{:.1f} mrad".format(unit_firsttraces/1e-3)
  frshelpers.plot.plot_bin_evolution(bin_centers/1e12, np.diff(bin_centers).mean()/4/1e12, data_beforeavg[:50,:].T/unit_firsttraces, corridor_label=corridor_label, ax=ax)
  ax.set_title("first traces")
  ax.set_ylabel("trace number")
  plt.tick_params(labelbottom=False, bottom=False) 

  ax = fig.add_subplot(gs[2,column], sharex=ax)
  corridor_label = "{:.1f}%".format(unit/1e-2) if column==0 else "{:.1f} mrad".format(unit/1e-3)
  frshelpers.plot.plot_bin_evolution(bin_centers/1e12, np.diff(bin_centers).mean()/4/1e12, data.T/unit, corridor_label=corridor_label, ax=ax, t=np.arange(data.shape[0])*average_traces)
  ax.set_ylabel("trace number")
  plt.tick_params(labelbottom=False, bottom=False) 
  ax.set_title("after {}-traces-average".format(average_traces))

  ax = fig.add_subplot(gs[3,column])
  frshelpers.plot.plot_allan(data.T, t_multiplier=average_traces, ax=ax)
  ax.grid(True, which="major", color="0.65")
  ax.grid(True, which="minor", color="0.85")
  ax.set_ylim(ylim)
  ax.set_xlim((1, data.shape[0]*average_traces))

  ax.set_xlabel("averaged traces")

  if column==0:
    ax.set_ylabel("relative amplitude precision")
    ax2 = ax.secondary_yaxis(1.01, functions=(
      functools.partial(frshelpers.plot.relamp_to_OD,OD_unit=1e-3),
      functools.partial(frshelpers.plot.OD_to_relamp,OD_unit=1e-3)
    ))
    ax2.set_ylabel("LOD (mOD)")
  else:
    ax.set_ylabel("phase precision (rad)")
    ax2 = ax.secondary_yaxis(1.01, functions=(
      functools.partial(frshelpers.plot.relamp_to_temporaljitter,nu=nu_center,t_unit=1e-18),
      functools.partial(frshelpers.plot.temporaljitter_to_relamp,nu=nu_center,t_unit=1e-18)
    ))
    ax2.set_ylabel("phase precision ({:.1f}-THz-as)".format(nu_center/1e12))

plt.savefig("allan_deviation.png")
plt.show()
