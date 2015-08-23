# QuaternionFourierTransform.jl

Discrete Quaternion Fourier Transform for Julia.

## Requirements

- [Quaternions.jl](https://github.com/peakbook/Quaternions.jl) (not official package)

## Example

```julia
using Quaternions
using QuaternionFourierTransform
using Images, TestImages

img = testimage("lena_color_256")
x = rand(Quaternion{Float64},4)

r = float(red(img.data))
g = float(green(img.data))
b = float(blue(img.data))
rezero = zeros(size(img))
qmat = quaternion(rezero, r, g, b)

qfreq = qft(qmat)
iqmat = iqft(qfreq)

# ...
```

You can specify the orthonormal basis like this.
```julia
mu = [quaternion(0,1,0,0),quaternion(0,0,1,0),quaternion(0,0,0,1)]
qft(m, mu)
```

## Reference

- Ell, T. A. and Sangwine, S. J., ``Hypercomplex Fourier Transforms of Color Images,'' IEEE Transactions on Image Processing, 16, (1), January 2007.
- Quaternion and octonion toolbox for Matlab, http://qtfm.sourceforge.net/

