---
title: 'Stingray 2: A fast and modern Python library for spectral timing'
tags:
- Python
- astronomy
- time series
- black holes
- neutron stars
authors:
- name: Matteo Bachetti
  orcid: 0000-0002-4576-9337
  affiliation: 1
- name: Daniela Huppenkothen
  orcid: 0000-0002-1169-7486
  affiliation: 2
- name: Abigail Stevens
  orcid: 0000-0002-5041-3079
  affiliation: 3
- name: John Swinbank
  affiliation: 4
- name: Guglielmo Mastroserio
  affiliation: 5
  orcid: 0000-0003-4216-7936
- name: Matteo Lucchini
  orcid: 0000-0002-2235-3347
  affiliation: 6
- name: Eleonora Veronica Lai
  affiliation: 1
  orcid: 0000-0002-6421-2198
- name: Johannes Buchner
  affiliation: 7
  orcid: 0000-0003-0426-6634
- name: Amogh Desai
  affiliation: 8
  orcid: 0000-0002-6015-9553
- name: Gaurav Joshi
  orcid: 0009-0009-2305-5008
  affiliation: 9
- name: Francesco Pisanu
  orcid: 0000-0003-0799-5760
  affiliation: 10
- name: Sri Guru Datta Pisupati
  orcid: 0009-0006-3745-5553
  affiliation: 11
- name: Swapnil Sharma
  affiliation: 12
- name: Mihir Tripathi
  affiliation: 13
  orcid: 0009-0007-2723-0315
- name: Dhruv Vats
  affiliation: 14
  orcid: 0009-0001-0542-0755

affiliations:
- name: INAF-Osservatorio Astronomico di Cagliari, via della Scienza 5, I-09047 Selargius (CA), Italy
  index: 1
- name: SRON
  index: 2
- name: MSU Museum, Michigan State University
  index: 3
- name: Dipartimento di Fisica, Universit`a Degli Studi di Milano, Via Celoria, 16, Milano, 20133, Italy
  index: 5
- name: Anton Pannekoek Institute, University of Amsterdam, Science Park 904, 1098 XH Amsterdam, The Netherlands
  index: 6
- name: Max Planck Institute for Extraterrestrial Physics, Giessenbachstrasse, 85741 Garching, Germany
  index: 7
- name: Carnegie Mellon University
  index: 8
- name: Indian Institute of Technology Gandhinagar
  index: 9
- name: LIPN-Université Sorbonne Paris Nord
  index: 10
- name: Chaitanya Bharathi Institute of Technology, Hyderabad,India
  index: 11
- name: Indian Institute of Technology, Mandi
  index: 12
- name: Academia Sinica Institute of Astronomy & Astrophysics 11F of Astronomy-Mathematics Building, AS/NTU, No. 1, Section 4, Roosevelt Road, Taipei 10617, Taiwan, R.O.C.
  index: 13
- name: Voltron Data, US
  index: 14

date: 01 October 2024
bibliography: joss.bib
aas-doi:
aas-journal:
---

# Summary

Time series analysis concerns the detection, characterization and modeling quantities that vary with time. The measured quantity may be anything from the average length of T-shirts in Southern Sardinia to the emission of light of our Sun.
This variability might be strictly periodic like a metronome, quasi-periodic like our heart beat, or stochastic, like the vibration of the ground during an earthquake.
Celestial objects are known to be change in brightness over time, driven by a diverse range of physical processes that include convection, stellar evolution and accretion of material onto black holes. Time scales range from sub-milliseconds to billions of years. Astrophysical research uses time series analysis to characterize the important time scales in the systems under scrutiny, and connect them to the underlying physical processes that cause them.
For example, the rotation of some pulsars, extremely dense stellar remnants, can be tracked over time and be considered almost like a cosmic clock. Other applications require complex modeling, including the study of the signals produced by the complicated interplay, propagation and partial re-emission of the light emitted by different regions around an accreting black hole. These studies require techniques that blend together traditional time series analysis and modeling of wavelength-dependent spectra [@uttley].

Stingray is an Astropy-affiliated [@astropy2013,@astropy2022] Python package that brings advanced timing techniques to the wider astronomical community, with a focus on high-energy astrophysics.
Stingray was previously described in [@stingrayjoss,@stingrayapj]. Its core functionality comprises Fourier-based analyses [@bachettihuppenkothen], but the package has expanded significantly over time in both scope and functionality. In this paper we describe the improvements to the software in the last ~5 years.

A core development goal has been to accelerate core stingray functionality, lower memory footprint, and refactor code to be extensive and interoperable, in order to prepare the library for the increasing size and complexity of modern astronomical datasets. Stingray’s core classes for Fourier analysis have shown dramatic increases in performance over time, as evident from [our benchmarks](https://stingray.science/stingray-benchmarks/). Stingray can now produce standard timing products (e.g. a Bartlett periodogram with a Nyquist frequency of 1000 Hz and a segment size of 128 s) of a typical high-flux NICER observation in ~one second. This is the result of algorithmic improvement, and of leveraging of Just-In-Time compilation through Numba in many key components of the code. A second improvement includes large-scale reorganization of the code to avoid duplication without major breaking changes to the API, and the creation of metaclasses that enable seamless integration with other popular array formats for time series (e.g. [Pandas](https://pandas.pydata.org/), [Xarray](https://docs.xarray.dev/en/stable/index.html), [Lightkurve](https://docs.lightkurve.org/), [Astropy Timeseries](https://docs.astropy.org/en/stable/timeseries/index.html)) and data formats ([FITS](), [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [extended CSV](https://docs.astropy.org/en/stable/io/ascii/ecsv.html)).

The originally planned implementation of spectral timing techniques — measures that combine Fourier analysis with spectral modeling - is now complete. Newly implemented techniques include the lag spectrum, covariance, rms, and coherence spectra. These methods are now showcased in extensive tutorials exploring NICER and NuSTAR observations.

We introduced a wide range of new techniques particularly designed to analyze unevenly sampled data sets, responding to the growing need for these techniques with the advent of large-scale astronomical time domain surveys, subject to irregular observing constraints. Methods include Gaussian Process modeling of quasi-periodic oscillations [hubner] and Lomb-Scargle cross spectra [scargle]. We have introduced the Fourier-Domain Acceleration Search [ransom] for pulsars; the H-test [dejager] and Phase Dispersion Minimization  [stellingwerf] statistics were also introduced into the pulsar sub package to evaluate the folded profiles of pulsars. We expanded the statistical capabilities of Stingray by introducing a number of statistical evaluation functions to estimate the statistics of periodograms, with particular attention to the upper limits on variable power.

Finally, we have added a number of high-level exploratory and diagnostic functionality specifically as an essential toolbox to characterize accreting compact objects during their outbursts: standard products such as color-color and hardness-intensity diagrams, and their equivalent diagnostics in the frequency domain, "power colors" [@heil].

In Stingray's design and development, we continue to strive to provide specific high-level functionality to the high-energy astrophysics community, but built on top of general-purpose classes and methods that are designed to be easily adapted and extended to other use cases. Ongoing work funded by the Italian [National Recovery and Resilience Plan](https://www.mef.gov.it/en/focus/The-National-Recovery-and-Resilience-Plan-NRRP/) is pushing Stingray's performance further with the use of GPUs and parallel computing in anticipation of large-scale astronomical time domain surveys for example with the Vera Rubin Telescope. In addition, the near-future will see an overhaul and redesign of Stingray's `modeling` subpackage in order to take advantage of recent developments in fast optimization and sampling algorithms and probabilistic programming. In order to facilitate spectral-timing with state-of-the-art instruments, we are actively working to integrate Stingray with ongoing software efforts improving modeling of astronomical spectra.

# Acknowledgments
MB and EVL are supported in part by Italian Research Center on High Performance Computing Big Data and Quantum Computing (ICSC), project funded by European Union - NextGenerationEU - and National
Recovery and Resilience Plan (NRRP) - Mission 4 Component 2 within the activities of Spoke 3
(Astrophysics and Cosmos Observations)
MB and GM were supported in part by PRIN TEC INAF 2019 ``SpecTemPolar! -- Timing analysis in the era of high-throughput photon detectors''
DH is supported by the Women In Science Excel (WISE) programme of the Netherlands Organisation for Scientific Research (NWO).
GM acknowledges financial support from the European Union’s Horizon Europe research and innovation programme under the Marie Sk\l{}odowska-Curie grant agreement No. 101107057
# References
