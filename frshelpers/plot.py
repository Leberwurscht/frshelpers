import numpy as np
import matplotlib.pyplot as plt

import allantools
import fourioso

relamp_to_relint = lambda relamp: 1-(1-relamp)**2
relint_to_relamp = lambda relint: 1-(1-relint)**.5
relint_to_OD = lambda relint, OD_unit: -np.log10(1 - relint)/OD_unit
OD_to_relint = lambda OD, OD_unit: 1 - 10**(-OD*OD_unit)
relamp_to_OD = lambda relamp, OD_unit: relint_to_OD(relamp_to_relint(relamp), OD_unit)
OD_to_relamp = lambda OD, OD_unit: relint_to_relamp(OD_to_relint(OD,OD_unit))

relamp_to_temporaljitter = lambda relamp, nu, t_unit: relamp/2/np.pi*nu**-1/t_unit
temporaljitter_to_relamp = lambda tempjitter, nu, t_unit: tempjitter*t_unit/nu**-1*2*np.pi

def spectral_phase_without_CEP_and_GD(nu, complex_spectrum, weights=None):
  """
    returns the spectral phase of `complex_spectrum` without GD and CEP contributions
  """

  # cancel GD
  nu_env = nu - nu[nu.size//2]
  t,At = fourioso.itransform(nu - nu[nu.size//2], complex_spectrum*weights)
  t0 = t[np.argmax(abs(At)**2)]
  complex_spectrum = complex_spectrum * np.exp(1j*2*np.pi*nu*t0)

  # cancel CEP
  phi0 = np.angle(np.average(complex_spectrum, weights=weights))
  complex_spectrum = complex_spectrum * np.exp(-1j*phi0)

  return np.angle(complex_spectrum)

def plot_bin_evolution(bin_centers, corridor_width, data, corridor_label=None, ax=None, plot_corridor=True, t=None, **kwargs):
  """
    plots the temporal evolution of a binned quantity along the y axis

    bin_centers (1d ndarray): defines the position along the x axis where each line is shown
    corridor_width (float): each line is surrounded by a corridor giving the scale - this defines the width of the corridor in x units
    data (ndarray with shape N_bins x N_t): 
    ax: matplotlib axes object to plot in (optional)
  """

  if ax is None: ax = plt.gca()
  if t is None: t = np.arange(data.shape[-1])

  colors = plt.cm.turbo_r(np.linspace(0,1,bin_centers.size))

  for bin_center, line_data, color, in zip(bin_centers, data, colors):
    if plot_corridor: s=ax.axvspan(bin_center-corridor_width/2, bin_center+corridor_width/2, alpha=0.2, color="b")
    ax.plot(bin_center+line_data*corridor_width, t, color=color, **kwargs)

  if not ax.yaxis_inverted(): ax.invert_yaxis()

  if plot_corridor:
    s.set_label(corridor_label)
    if corridor_label is not None: ax.legend(loc=1, framealpha=1)

def allandeviation(data, overlapping=False, return_error=False):
  func = allantools.oadev if overlapping else allantools.adev
  taus = np.logspace(0, np.log10(data.size/2), 200)
  t, ad, ad_error, _ = func(data, rate=1, data_type="freq", taus=taus)
  corridor_lower, corridor_upper = ad-ad_error, ad+ad_error

  if return_error:
    return t, ad, corridor_lower, corridor_upper
  else:
    return t, ad

def plot_allan(data, ax=None, t_multiplier=1, **kwargs):
  if ax is None: ax = plt.gca()

  colors = plt.cm.turbo(np.linspace(0,1,data.shape[0]))

  for line_data, color in zip(data, colors):
    t,ad,corridor_lower,corridor_upper = allandeviation(line_data, return_error=True)
    t = t * t_multiplier

    ax.fill_between(t, corridor_lower, corridor_upper, alpha=0.2, color=color, **kwargs)
    ax.plot(t, ad, color=color, **kwargs)

  ax.set_xscale("log")
  ax.set_yscale("log")
