module QuaternionFourierTransform

using Quaternions
export qft, iqft, qconv2
const defaultbasis = quaternion(0,1,1,1)/sqrt(3)

function qft{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}; mu::Union(Quaternion{S},NTuple{3,Quaternion{S}})=defaultbasis, LR::Bool=false)
    mus = orthonormal_basis(mu...)
    if LR # false: Left, true: Right
        qft_core(x, fft, mus, one(T))
    else
        qft_core(x, fft, mus, -one(T))
    end
end

function iqft{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}; mu::Union(Quaternion{S},NTuple{3,Quaternion{S}})=defaultbasis, LR::Bool=false)
    mus = orthonormal_basis(mu...)
    if LR # false: Left, true: Right
        qft_core(x, ifft, mus, one(T))
    else  
        qft_core(x, ifft, mus, -one(T))
    end
end

function qft_core{T<:Real,S<:Real}(x::AbstractArray{Quaternion{T}}, ft::Function, mus::NTuple{3,Quaternion{S}}, s::T)
    a, b, c, d = change_basis_core(x, mus)
    fc1, fc2 = ft(complex(a, b)), ft(complex(c, s*d))
    change_basis_core(real(fc1), imag(fc1), real(fc2), s*imag(fc2), mus)
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

function change_basis{T<:Real}(x::AbstractArray{Quaternion{T}}, mus::NTuple{3,Quaternion{T}}, inverse::Bool=false)
    if inverse
        change_basis_core(real(x),imagi(x),imagj(x),imagk(x),mus)
    else
        quaternion(change_basis_core(x,mus)...)
    end
end

function change_basis_core{T<:Real}(x::AbstractArray{Quaternion{T}}, mus::NTuple{3,Quaternion{T}})
    x_im = imag(x)
    return real(x), -real(map(x->x|mus[1],x_im)), -real(map(x->x|mus[2],x_im)), -real(map(x->x|mus[3],x_im))
end

function change_basis_core{T<:Real,S<:Real}(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, d::AbstractArray{T}, mus::NTuple{3,Quaternion{S}})
    return a + b*mus[1] + c*mus[2] + d*mus[3]
end

function qconv2{T<:Real,S<:Real}(A::AbstractArray{Quaternion{T}}, B::AbstractArray{Quaternion{T}}; mu::Union(Quaternion{S},NTuple{3,Quaternion{S}})=defaultbasis, LR::Bool=false)
    mus = orthonormal_basis(mu...)
    sa, sb = size(A), size(B)
    At = zeros(Quaternion{T}, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
    Bt = zeros(Quaternion{T}, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
    At[1:sa[1], 1:sa[2]] = A
    Bt[1:sb[1], 1:sb[2]] = B

    fC = LR ? qconv2_r(At,Bt,mus) : qconv2_l(At,Bt,mus)
    return iqft(fC,mu=mus)
end

function qconv2_l(At,Bt,mus)
    Aa, Ab, Ac, Ad = change_basis_core(At, mus)
    Bpa, Bpb, Bpc, Bpd = change_basis_core(Bt, mus)

    CA1, CA2 = complex(Aa, Ab), complex(Ac, -Ad)
    CBp1, CBp2 = complex(Bpa, Bpb), complex(Bpc, Bpd)
    p = plan_fft(CA1)
    fCA1, fCA2 = p(CA1), p(CA2)
    fCBp1, fCBp2 = p(CBp1), p(CBp2)

    fBp = change_basis_core(real(fCBp1), imag(fCBp1), real(fCBp2), imag(fCBp2), mus)

    return ((real(fCA1)+imag(fCA1)*mus[1]) .* fBp) + ((real(fCA2)-imag(fCA2)*mus[1])*mus[2] .* fBp)
end

function qconv2_r(At,Bt,mus)
    Ba, Bb, Bc, Bd = change_basis_core(Bt, mus)
    Apa, Apb, Apc, Apd = change_basis_core(At, mus)

    CB1, CB2 = complex(Ba, Bb), complex(Bc, -Bd)
    CAp1, CAp2 = complex(Apa, Apb), complex(Apc, -Apd)
    p = plan_fft(CB1)
    fCB1, fCB2 = p(CB1), p(CB2)
    fCAp1, fCAp2 = p(CAp1), p(CAp2)

    fAp = change_basis_core(real(fCAp1), imag(fCAp1), real(fCAp2), -imag(fCAp2), mus)

    return (fAp .* (real(fCB1)+imag(fCB1)*mus[1])) + (fAp .* ((real(fCB2)-imag(fCB2)*mus[1])*mus[2]))
end

end # module
