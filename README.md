# QuaternionFourierTransform.jl

[![Build Status](https://travis-ci.org/peakbook/QuaternionFourierTransform.jl.svg?branch=master)](https://travis-ci.org/peakbook/QuaternionFourierTransform.jl)
[![Coverage Status](https://coveralls.io/repos/peakbook/QuaternionFourierTransform.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/peakbook/QuaternionFourierTransform.jl?branch=master)

Discrete Quaternion Fourier Transform for Julia.

## Requirements

- [Quaternions.jl](https://github.com/peakbook/Quaternions.jl) (not official package)

## Installation
```julia
pkg> add https://github.com/peakbook/Quaternions.jl
pkg> add https://github.com/peakbook/QuaternionFourierTransform.jl
```

## Example

```julia
using Quaternions
using QuaternionFourierTransform

qmat = imag(rand(QuaternionF64, 10, 10))

# QFT
qfreq = qft(qmat)

# IQFT
iqmat = iqft(qfreq)

# Convolution
qfilter = ...
qconv(qmat, qfilter)
```


You can specify the orthonormal basis as follows:
```julia
# Default value is `QuaternionFourierTransform.defaultbasis`
q = (quaternion(0,1,0,0),quaternion(0,0,1,0),quaternion(0,0,0,1))
qft(m, mu=q)
```

See [example](./example/ex.jl).

## Reference

- Ell, T. A. and Sangwine, S. J., ``Hypercomplex Fourier Transforms of Color Images,'' IEEE Transactions on Image Processing, 16, (1), January 2007.
- Quaternion and octonion toolbox for Matlab, http://qtfm.sourceforge.net/

