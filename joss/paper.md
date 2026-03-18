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
  affiliation: 2
- name: Eleonora Veronica Lai
  affiliation: 1
  orcid: 0000-0002-6421-2198
- name: Johannes Buchner
  affiliation: 6
  orcid: 0000-0003-0426-6634
- name: Amogh Desai
  affiliation: 7
  orcid: 0000-0002-6015-9553
- name: Gaurav Joshi
  orcid: 0009-0009-2305-5008
  affiliation: 8
- name: Francesco Pisanu
  orcid: 0000-0003-0799-5760
  affiliation: 9
- name: Sri Guru Datta Pisupati
  orcid: 0009-0006-3745-5553
  affiliation: 10
- name: Swapnil Sharma
  affiliation: 11
- name: Mihir Tripathi
  affiliation: 12
  orcid: 0009-0007-2723-0315
- name: Dhruv Vats
  affiliation: 13
  orcid: 0009-0001-0542-0755


affiliations:
- name: INAF-OACagliari, Italy
  index: 1
- name: UvA, The Netherlands
  index: 2
- name: MSU Museum, Michigan State University, USA
  index: 3
- name: ASTRON, The Netherlands
  index: 4
- name: Dip. Fisica, Università di Milano, Italy
  index: 5
- name: MPE, Garching, Germany
  index: 6
- name: Carnegie Mellon University, USA
  index: 7
- name: Indian Institute of Technology Gandhinagar, India
  index: 8
- name: LIPN-Université Sorbonne Paris Nord, France
  index: 9
- name: Chaitanya Bharathi Institute of Technology, Hyderabad, India
  index: 10
- name: Indian Institute of Technology, Mandi, India
  index: 11
- name: Academia Sinica Institute of Astronomy & Astrophysics, Taipei, Taiwan, R.O.C.
  index: 12
- name: Voltron Data, USA
  index: 13

date: 01 October 2024
bibliography: joss.bib
---

# Summary

Stingray is an Astropy-affiliated [@astropy2013; @astropy2022] Python package that brings advanced timing techniques to the wider astronomical community, with a focus on high-energy astrophysics, but built on top of general-purpose classes and methods that are designed to be easily adapted and extended to other use cases.
Stingray was previously described by @stingrayapj and @stingrayjoss. Its core functionality comprises Fourier-based analyses [@bachettihuppenkothen], but the package has expanded significantly over time in both scope and functionality. In this paper we describe the improvements to the software in the last ~5 years.

# Background

Time series analysis concerns the detection, characterization and modeling quantities that vary with time.
This variability might be strictly periodic like a metronome, quasi-periodic like our heart beat, or stochastic, like the vibration of the ground during an earthquake.
Celestial objects are known to be change in brightness over time, driven by a diverse range of physical processes. Time scales range from sub-milliseconds to billions of years.
For example, the rotation of some pulsars, extremely dense stellar remnants, can be tracked over time and be considered almost like a cosmic clock. Other applications require complex modeling, including the study of the signals produced by the complicated interplay, propagation and partial re-emission of the light emitted by different regions around an accreting black hole. These studies require techniques that blend together traditional time series analysis and modeling of wavelength-dependent spectra [@uttley; @bachettihuppenkothen].

# Statement of need

Until 2015, the techniques described above were used by competing groups using their own in-house codes. Very few of them were shared publicly, often with poor documentation and/or based on commercial or niche programming languages. Stingray brought them to the general astronomical community, and is now used worldwide, especially by young students.

# Five years of Development

A core development goal has been to accelerate core Stingray functionality, lower the memory footprint, and refactor the code to be extensive and interoperable.
Stingray's core classes have shown dramatic increase in performance over time, as evident from [our benchmarks](https://stingray.science/stingray-benchmarks/). Stingray can now produce standard timing products
of a typical high-flux NICER observation in roughly one second. This is thanks to algorithmic improvement, and Just-In-Time compilation through Numba of many key components of the code. We reorganized the code to avoid duplication,
and created metaclasses that enable seamless integration with other popular array formats for time series (e.g. [Pandas](https://pandas.pydata.org/), [Lightkurve](https://docs.lightkurve.org/), [Astropy Timeseries](https://docs.astropy.org/en/stable/timeseries/index.html)) and data formats ([FITS](), [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [extended CSV](https://docs.astropy.org/en/stable/io/ascii/ecsv.html)).

We completed the originally planned implementation of spectral timing techniques. Newly implemented techniques include the lag spectrum, covariance, rms, and coherence spectra. These methods are now showcased in extensive tutorials exploring NICER and NuSTAR observations.

We introduced a wide range of new techniques designed to analyze unevenly sampled data sets, responding to the growing need for these techniques from astronomical time domain surveys, subject to irregular observing constraints. Methods include Gaussian Process modeling of quasi-periodic oscillations [@hubner] and Lomb-Scargle cross spectra [@scargle]. We have introduced the Fourier-Domain Acceleration Search [@ransom], the H-test [@dejager] and Phase Dispersion Minimization  [@stellingwerf] statistics into the pulsar subpackage.
We expanded the statistical capabilities of Stingray,
with particular attention to the calculation of confidence limits and upper limits on variability measures.

Finally, we have added high-level exploratory and diagnostic functionality, such as color-color and hardness-intensity diagrams, and their equivalent diagnostics in the frequency domain, "power colors" [@heil].

Ongoing work funded by the Italian [National Recovery and Resilience Plan](https://www.mef.gov.it/en/focus/The-National-Recovery-and-Resilience-Plan-NRRP/) is pushing Stingray's performance further with the use of GPUs and parallel computing in anticipation of large-scale astronomical time domain surveys for example with the Vera Rubin Telescope. In addition, the near future will see an overhaul and redesign of Stingray's `modeling` subpackage in order to take advantage of recent developments in fast optimization and sampling algorithms and probabilistic programming. In order to facilitate spectral timing with state-of-the-art instruments, we are actively working to integrate Stingray with ongoing software efforts improving modeling of astronomical spectra.

# Acknowledgments

MB and EVL are supported in part by Italian Research Center on High Performance Computing Big Data and Quantum Computing (ICSC) project funded by European Union - NextGenerationEU - and National
Recovery and Resilience Plan (NRRP) - Mission 4 Component 2 within the activities of Spoke 3
(Astrophysics and Cosmos Observations).
MB and GM were supported in part by PRIN TEC INAF 2019 ``SpecTemPolar! -- Timing analysis in the era of high-throughput photon detectors''.
DH is supported by the Women In Science Excel (WISE) programme of the Netherlands Organisation for Scientific Research (NWO).
GM acknowledges financial support from the European Union's Horizon Europe research and innovation programme under the Marie Sk\l{}odowska-Curie grant agreement No. 101107057.

# References
