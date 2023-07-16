# EFTofLSSFisher: Fisher forecasts with the EFTofLSS
[![](https://img.shields.io/badge/arXiv-2307.04992%20-red.svg)](https://arxiv.org/abs/2307.04992)
[![](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/YDonath/EFTofLSSFisher/LICENSE](https://github.com/YDonath/EFTofLSSFisher/blob/main/LICENSE))
## General info
EFTofLSSFisher is a code written in Mathematica and Python, designed to forecast cosmological constraints from future cosmological surveys using the power spectrum and bispectrum of biased tracers in redshift space at the one loop order. The equations it is based on can be found in  [arXiv:2211.17130](https://arxiv.org/abs/2211.17130), and the integration techniques are from [arXiv:2212.07421](https://arxiv.org/abs/2212.07421) and the code is the [IntegerInt](https://github.com/dbraganca/python-integer-powers), for which we include the setup files. 

Details about the procedure and results are in [arXiv:2307.04992
](https://arxiv.org/abs/2307.04992)

## Dependencies
Apart from pip-installable numerical libraries like [NumPy](https://numpy.org/) and [SciPy](http://scipy.org/), linear power spectra and transfer functions are computed with [CLASS](https://lesgourg.github.io/class_public/class.html).

## Getting started (constraints only)
We include full Fisher matrices for BOSS, DESI and MegaMapper.  To obtain constraints in various parameter combinations run the notebook `CreatePlots/Fisher_Plots.ipynb`.
## Getting started (Running forecasts)
To run the full Fisher forecast, first compile the code to generate bispectrum loops. Compile `Createjmats/setup.py` by running
```
python setup.py build_ext --inplace
```
and  `Createjmats/setup.py` by running 
```
python setup_jfunc.py build_ext --inplace
```
see the original [release](https://github.com/dbraganca/python-integer-powers) for more details.

Then generate the bispectrum loops using the `Createjmats/computejmats.ipynb` notebook. Any changes in triangle configurations can be done here.
Finally run and export Fisher information matrices in the `EFTofLSSFisher.nb` Mathematica notebook. 
(optional) Any changes in the cosmological parameters can be made by rerunning the linear powers pectra in `Plin/generate_plin_and_transfer.ipynb`.
