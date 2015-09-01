module QuaternionFourierTransform

using Quaternions
export qft, iqft
const defaultbasis = quaternion(0,1,1,1)/sqrt(3)

function qft{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}; mu::Union(Quaternion{S},NTuple{3,Quaternion{S}})=defaultbasis, LR::Bool=false)
    mus = orthonormal_basis(mu...)
    if LR # false: Left, true: Right
        qft_r(x,fft,mus)
    else
        qft_l(x,fft,mus)
    end
end

function iqft{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}; mu::Union(Quaternion{S},NTuple{3,Quaternion{S}})=defaultbasis, LR::Bool=false)
    mus = orthonormal_basis(mu...)
    if LR # false: Left, true: Right
        qft_r(x,ifft,mus)
    else  
        qft_l(x,ifft,mus)
    end
end

function qft_l{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}, ft::Function, mus::NTuple{3,Quaternion{S}})
    a, b, c, d = trans_l(x, mus)
    fc1, fc2 = ft(complex(a, b)), ft(complex(c, d))
    itrans_l(real(fc1), imag(fc1), real(fc2), imag(fc2), mus)
end

function qft_r{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}, ft::Function, mus::NTuple{3,Quaternion{S}})
    a, b, c, d = trans_r(x, mus)
    fc1, fc2 = ft(complex(a, b)), ft(complex(c, d))
    itrans_r(real(fc1), imag(fc1), real(fc2), imag(fc2), mus)
end

function orthonormal_basis(m1::Quaternion)
    m2 = imag(rand(typeof(m1)))
    m3 = imag(m2/m1)
    m2 = imag(m3/m1)
    m1 /= norm(m1)
    m2 /= norm(m2)
    m3 /= norm(m3)
    return orthonormal_basis(m1,m3,m2)
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

    return (m1,m2,m3)
end

function trans_l{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}, mus::NTuple{3,Quaternion{S}})
    x_im = imag(x)
    return real(x), -real(map(x->x|mus[1],x_im)), -real(map(x->x|mus[2],x_im)), -real(map(x->x|mus[3],x_im))
end

function itrans_l{T<:Real,S<:Real}(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, d::AbstractArray{T}, mus::NTuple{3,Quaternion{S}})
    return a + b*mus[1] + c*mus[2] + d*mus[3]
end

function trans_r{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}, mus::NTuple{3,Quaternion{S}})
    x_im = imag(x)
    return real(x), -real(map(x->x|mus[1],x_im)), -real(map(x->x|mus[3],x_im)), -real(map(x->x|mus[2],x_im))
end

function itrans_r{T<:Real,S<:Real}(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, d::AbstractArray{T}, mus::NTuple{3,Quaternion{S}})
    return a + b*mus[1] + c*mus[3] + d*mus[2]
end

end # module
