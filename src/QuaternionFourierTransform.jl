module QuaternionFourierTransform

using Quaternions
export qft, iqft
const defaultbasis = quaternion(0,1,1,1)/sqrt(3)

function qft{T<:Real}(x::AbstractArray{Quaternion{T}}, mu=defaultbasis, LR=false::Bool)
    mus = orthonormal_basis(mu...)
    if LR # false: Left, true: Right
        qft_r(x,fft,mus)
    else
        qft_l(x,fft,mus)
    end
end

function iqft{T<:Real}(x::AbstractArray{Quaternion{T}}, mu=defaultbasis, LR=false::Bool)
    mus = orthonormal_basis(mu...)
    if LR # false: Left, true: Right
        qft_r(x,ifft,mus)
    else  
        qft_l(x,ifft,mus)
    end
end

function qft_l{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}, ft::Function, mus::AbstractArray{Quaternion{S}})
    x_im = imag(x)
    a_ = real(x)
    b_ = -real(map(x->x|mus[1],x_im))
    c_ = -real(map(x->x|mus[2],x_im))
    d_ = -real(map(x->x|mus[3],x_im))

    c1 = complex(a_, b_)
    c2 = complex(c_, d_)
    fc1 = ft(c1)
    fc2 = ft(c2)

    x_t = real(fc1) + imag(fc1)*mus[1] + real(fc2)*mus[2] + imag(fc2)*mus[3]
    a = real(x_t)
    b = imagi(x_t)
    c = imagj(x_t)
    d = imagk(x_t)

    return quaternion(a,b,c,d)
end

function qft_r{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}, ft::Function, mus::AbstractArray{Quaternion{S}})
    x_im = imag(x)
    a_ = real(x)
    b_ = -real(map(x->x|mus[1],x_im))
    c_ = -real(map(x->x|mus[3],x_im))
    d_ = -real(map(x->x|mus[2],x_im))

    c1 = complex(a_, b_)
    c2 = complex(c_, d_)
    fc1 = ft(c1)
    fc2 = ft(c2)

    x_t = real(fc1) + imag(fc1)*mus[1] + real(fc2)*mus[3] + imag(fc2)*mus[2]
    a = real(x_t)
    b = imagi(x_t)
    c = imagj(x_t)
    d = imagk(x_t)

    return quaternion(a,b,c,d)
end

function orthonormal_basis(m1::Quaternion)
    m2 = imag(rand(typeof(m1)))
    m3 = imag(m2/m1)
    m2 = imag(m3/m1)
    m1 /= norm(m1)
    m2 /= norm(m2)
    m3 /= norm(m3)
    return orthonormal_basis(m1,m2,m3)
end

function orthonormal_basis(m1::Quaternion, m2::Quaternion, m3::Quaternion)
    @assert(
    real(m1) == zero(typeof(real(m1))) &&
    real(m2) == zero(typeof(real(m2))) &&
    real(m3) == zero(typeof(real(m3))), "The transform axis must be a pure quaternion.")

    m = reshape([imagi(m1),imagj(m1),imagk(m1),
                 imagi(m2),imagj(m2),imagk(m2),
                 imagi(m3),imagj(m3),imagk(m3)],(3,3))
    
    if maximum(m*m'-eye(3))>10*eps() 
        warn("The basis matrix is not accurately orthogonal.")
    end

    return [m1,m2,m3]
end

function trans_l{T<:Real}(x::AbstractArray{Quaternion{T}}, mus)
    mus = orthonormal_basis(mus...)
    x_im = imag(x)
    return quaternion(real(x), -real(map(x->x|mus[1],x_im)), -real(map(x->x|mus[2],x_im)), -real(map(x->x|mus[3],x_im)))
end

function itrans_l{T<:Real}(x::AbstractArray{Quaternion{T}}, mus)
    mus = orthonormal_basis(mus...)
    return real(x) + imagi(x)*mus[1] + imagj(x)*mus[2] + imagk(x)*mus[3]
end

function trans_r{T<:Real}(x::AbstractArray{Quaternion{T}}, mus)
    mus = orthonormal_basis(mus...)
    x_im = imag(x)
    return quaternion(real(x), -real(map(x->x|mus[1],x_im)), -real(map(x->x|mus[3],x_im)), -real(map(x->x|mus[2],x_im)))
end

function itrans_r{T<:Real}(x::AbstractArray{Quaternion{T}}, mus)
    mus = orthonormal_basis(mus...)
    return real(x) + imagi(x)*mus[1] + imagj(x)*mus[3] + imagk(x)*mus[2]
end

end # module
