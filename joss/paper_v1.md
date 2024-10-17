---
title: 'stingray: A modern Python library for spectral timing'
tags:
- Python
- astronomy
- time series
- black holes
- neutron stars
authors:
- name: Daniela Huppenkothen
  orcid: 0000-0002-1169-7486
  affiliation: 1
- name: Matteo Bachetti
  orcid: 0000-0002-4576-9337
  affiliation: 2
- name: Abigail Stevens
  orcid: 0000-0002-5041-3079
  affiliation: "3, 4"
- name: Simone Migliari
  affiliation: "5, 6"
- name: Paul Balm
  affiliation: 6
- name: Omar Hammad
  affiliation: 7
- name: Usman Mahmood Khan
  affiliation: 8
- name: Himanshu Mishra
  affiliation: 9
- name: Haroon Rashid
  affiliation: 10
- name: Swapnil Sharma
  affiliation: 11
- name: Evandro Martinez Ribeiro
  affiliation: 12
- name: Ricardo Valles Blanco
  affiliation: 6
affiliations:
- name: DIRAC Institute, Department of Astronomy, University of Washington, 3910 15th Ave NE, Seattle, WA 98195
  index: 1
- name: INAF-Osservatorio Astronomico di Cagliari, via della Scienza 5, I-09047 Selargius (CA), Italy
  index: 2
- name: Department of Physics & Astronomy, Michigan State University, 567 Wilson Road, East Lansing, MI 48824, USA
  index: 3
- name: Department of Astronomy, University of Michigan, 1085 South University Avenue, Ann Arbor, MI 48109, USA
  index: 4
- name: ESAC/ESA, XMM-Newton Science Operations Centre, Camino Bajo del Castillo s/n, Urb. Villafranca del Castillo, 28692, Villanueva de la Caada, Madrid, Spain
  index: 5
- name: Timelab Technologies Ltd., 20-22 Wenlock Road, London N1 7GU, United Kingdom
  index: 6
- name: AinShams University, Egypt
  index: 7
- name: Department of Computer Science, North Carolina State University, Raleigh, USA
  index: 8
- name: Indian Institute of Technology, Kharagpur West Bengal, India 721302
  index: 9
- name: National University of Sciences and Technology (NUST), Islamabad 44000, Pakistan
  index: 10
- name: Indian Institute of Technology Mandi, Mandi, Himachal Pradesh, India
  index: 11
- name: Kapteyn Astronomical Institute, University of Groningen, P.O. Box 800, NL-9700 AV Groningen, The Netherlands
  index: 12
date: 10 June 2019
bibliography: joss.bib
aas-doi: 10.3847/1538-4357/ab258d
aas-journal: Astrophysical Journal
---

# Summary

Many celestial objects vary in brightness on timescales of milliseconds to centuries. These ``light curves''--variations of brightness of an object as a function of time--often encode interesting physical processes that can help us learn about the nature of the celestial bodies that produced them.
In stars like our sun, typical time scales tell us about stellar rotation, starspots and internal physics like convection. In remnants of stellar explosions like neutron stars, we can use time series to learn about the densest matter known in the universe. Finally, variations in brightness of radiation emitted by gas falling into a black hole give important clues to the nature of gravity and allow us to test General Relativity to high precision.
Unravelling the underlying physical processes requires sophisticated statistical and signal processing methods, largely based on Fourier analysis, now well-established in this field.

Stingray is an Astropy-affiliated [@astropy, @astropy2] Python package, making a large range of routinely used time series analysis methods available to the astronomy community. It is based on existing implementations of Fourier-space methods in Numpy [@numpy] and Scipy [@scipy], but conveniently wraps them in classes and functions that allow easy application on astronomical data sets, especially from X-ray timing telescopes like the Rossi X-ray Timing Explorer (RXTE) [@Bradtetal93], the Nuclear Spectroscopic Telescope Array (NuSTAR)[@nustar13] and the Neutron Star Interior Composition Explorer (NICER) [@gendreau2016].  
Stingray is a modular, class-based library aiming to allow users to build custom workflows for their particular data set and science problem, and implements common operations such as periodograms with standard normalizations, cross spectra and coherence, auto- and cross-correlations, as well as higher-order Fourier products such as bispectra and spectral-timing methods like covariance spectra. The latter consider both time and wavelength of the arriving radiation simultaneously and allows for more comprehensive studies of the underlying physical system. Stingray also implements submodules that allow efficient parametric modelling of periodograms, simulations of realistic time series, and specialized tools to study pulsars.

Stingray was designed with a flexible and extensible API to be end-user friendly, but also lies at the core of two other packages: HENDRICS [@hendrics], which implements end-to-end versions of standard data analysis workflows, and DAVE, a graphical user interface designed to enable high-level exploratory data analysis on astronomical time series. A longer publication on the underlying methodology and implementation can be found in [@2019arXiv190107681H], while the source code itself is available on GitHub [@stingraysoftware] as part of a larger ecosystem including tutorials, as well as the repositories for HENDRICS and DAVE.


# Acknowledgments
D.H. acknowledges support from the DIRAC Institute in the Department of Astronomy at the University of Washington. The DIRAC Institute is supported through generous gifts from the Charles and Lisa Simonyi Fund for Arts and Sciences, and the Washington Research Foundation. M.B. is supported in part by the Italian Space Agency through agreement ASI-INAF n.2017-12-H.0 and ASI-INFN agreement n.2017-13-H.0. A.L.S. is supported by an NSF Astronomy and Astrophysics Postdoctoral Fellowship under award AST1801792. S.S. was supported by Google Summer of Code 2018. We thank Astro Hack Week for providing the venue that started this project and the Lorentz Center workshop ‘The X- ray Spectral-Timing Revolution’ (February 2016) that started the collaboration. We thank the Google Summer of Code Program for funding a total 6 students who implemented a large fraction of the various library components over three summers.

# References
