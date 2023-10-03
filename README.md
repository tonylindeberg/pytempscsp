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
in tempscsp.py.

## References

Lindeberg (2023) "A time-causal and time-recursive temporal scale-space representation
of temporal signals and past time", Biological Cybernetics 117 (1-2): 21-59.
([Open Access](http://dx.doi.org/10.1007/s00422-022-00953-6))

Lindeberg (2016) "Time-causal and time-recursive spatio-temporal receptive fields",
Journal of Mathematical Imaging and Vision 55(1): 50-88.
([Open Access](https://doi.org/10.1007/s10851-015-0613-9))

The time-causal limit kernel was first defined in Lindeberg (2016), however,
then also in combination with a spatial domain, and experimentally tested on
video data. The later overview paper (Lindeberg 2023) gives a dedicated treatment
for a purely temporal domain, and also with relations to Koenderink's scale-time
kernels and the ex-Gaussian kernel.

