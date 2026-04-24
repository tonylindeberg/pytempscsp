# pytempscsp : Temporal Scale Space Toolbox for Python

For performing temporal smoothing with the time-causal limit kernel and
for computing discrete temporal derivative approximations by applying
temporal difference operators to the smoothed data.

This code is the result of porting a subset of the routines in the Matlab
package tempscsp to Python, however, with different interfaces for the functions.

For examples of how to apply these functions for smoothing temporal signals
to different temporal scales in a fully time-causal manner, please see the
enclosed Jupyter notebook [tempscspdemo.ipynb](https://github.com/tonylindeberg/pytempscsp/blob/main/tempscspdemo.ipynb).

For more technical descriptions about the respective functions, please see
the documentation strings for the respective functions in the source code
in [tempscsp.py](https://github.com/tonylindeberg/pytempscsp/blob/main/pytempscsp/tempscsp.py).

## Time-causal temporal derivatives and bandpass wavelets

For examples of how to use extensions of this code for computing
time-causal temporal derivatives, please see the enclosed Jupyter
notebooks
[sineexpwave.ipynb](https://github.com/tonylindeberg/pytempscsp/blob/main/sineexpwave.ipynb),
[vibration.ipynb](https://github.com/tonylindeberg/pytempscsp/blob/main/vibration.ipynb)
and
[cymbal44.ipynb](https://github.com/tonylindeberg/pytempscsp/blob/main/cymbal44.ipynb).

For examples of how to use extensions of this code for computing
bandpass wavelets based on either 
(i) differences-of-time-causal-limit-kernels (DoT),
(ii) differences-of-Gaussians-kernels (DoG) or
(iii) differences-of-exponentials (DoE), 
please see the enclosed Jupyter
notebooks
[bandpass-blocks.ipynb](https://github.com/tonylindeberg/pytempscsp/blob/main/bandpass-blocks.ipynb)
and
[bandpass-riemann.ipynb](https://github.com/tonylindeberg/pytempscsp/blob/main/bandpass-riemann.ipynb).

For more technical descriptions about the respective functions, please see
the documentation strings for the respective functions in the source code
in
[tempscsel.py](https://github.com/tonylindeberg/pytempscsp/blob/main/pytempscsp/tempscsel.py)
and
[bandpass.py](https://github.com/tonylindeberg/pytempscsp/blob/main/pytempscsp/bandpass.py).

## Dependencies

Some parts of the code depend on the pyscsp library available at
[GitHub](https://github.com/tonylindeberg/pyscsp).

The Jupyter notebooks for computing visualizations on bandpass
wavelets depend on a function in the PyWavelets library for generating
the input data.

## Installation

This package is available 
through pip and can installed by

```bash
pip install pytempscsp
```

This package can also be downloaded directly from GitHub:

```bash
git clone git@github.com:tonylindeberg/pytempscsp.git
```

## References

Lindeberg (2023) "A time-causal and time-recursive temporal scale-space representation
of temporal signals and past time", Biological Cybernetics 117 (1-2): 21-59.
([Open Access](http://dx.doi.org/10.1007/s00422-022-00953-6))

Lindeberg (2023) "Time-causal and time-recursive wavelets", Frontiers
in Signal Processing, to appear, preprint at  arXiv:2510.05834.
([Open Access](https://doi.org/10.48550/arXiv.2510.05834))

Lindeberg (2016) "Time-causal and time-recursive spatio-temporal receptive fields",
Journal of Mathematical Imaging and Vision 55(1): 50-88.
([Open Access](https://doi.org/10.1007/s10851-015-0613-9))

The time-causal limit kernel was first defined in Lindeberg (2016), however,
then also in combination with a spatial domain, and experimentally tested on
video data. The later overview paper (Lindeberg 2023) gives a dedicated treatment
for a purely temporal domain, and also with relations to Koenderink's scale-time
kernels and the ex-Gaussian kernel. 

The extension to time-causal temporal derivatives was first performed 
in Lindeberg (2016), additionally considered in Lindeberg (2023) and
then extended to bandpass wavelets and more general time-causal
wavelets in Lindeberg (2025).
