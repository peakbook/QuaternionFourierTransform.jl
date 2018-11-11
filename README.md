# QuaternionFourierTransform.jl

Discrete Quaternion Fourier Transform for Julia.

## Requirements

- [Quaternions.jl](https://github.com/peakbook/Quaternions.jl) (not official package)

## Example

```julia
using Quaternions
using QuaternionFourierTransform

qmat = imag(rand(Quaternion{Float64},10,10))
qfreq = qft(qmat)
iqmat = iqft(qfreq)

# ...
```


You can specify the orthonormal basis like this. Default is `QuaternionFourierTransform.defaultbasis`
```julia
q = (quaternion(0,1,0,0),quaternion(0,0,1,0),quaternion(0,0,0,1))
qft(m, mu=q)
```

See [example](./example/ex.jl).

## Reference

- Ell, T. A. and Sangwine, S. J., ``Hypercomplex Fourier Transforms of Color Images,'' IEEE Transactions on Image Processing, 16, (1), January 2007.
- Quaternion and octonion toolbox for Matlab, http://qtfm.sourceforge.net/

