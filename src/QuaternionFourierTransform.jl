__precompile__()

module QuaternionFourierTransform

using Quaternions
export qft, iqft, qconv
const defaultbasis = Quaternion(0,1,1,1)/sqrt(3)
getbasis(T::DataType) = convert(T, defaultbasis)

function qft{T<:Quaternion}(x::AbstractArray{T}; mu::Union{T,NTuple{3,T}}=getbasis(eltype(x)), LR::Symbol=:left)
    mus = orthonormal_basis(mu...)
    if LR == :left
        qft_core(x, fft, mus, 1)
    elseif LR == :right
        qft_core(x, fft, mus, -1)
    else
        error("LR must be :left or :right.")
    end
end

function iqft{T<:Quaternion}(x::AbstractArray{T}; mu::Union{T,NTuple{3,T}}=getbasis(eltype(x)), LR::Symbol=:left)
    mus = orthonormal_basis(mu...)
    if LR == :left
        qft_core(x, ifft, mus, 1)
    elseif LR == :right  
        qft_core(x, ifft, mus, -1)
    else
        error("LR must be :left or :right.")
    end
end

function qft_core{T<:Quaternion}(x::AbstractArray{T}, ft::Function, mus::NTuple{3,T}, s::Integer)
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
    
    if maximum(m*m'-eye(3))>10*eps(real(eltype(m))) 
        warn("The basis matrix is not accurately orthogonal.")
    end

    return (m1,m2,m3)
end

function change_basis{T<:Quaternion}(x::AbstractArray{T}, mus::NTuple{3,T}, inverse::Bool=false)
    if inverse
        change_basis_core(real(x),imagi(x),imagj(x),imagk(x),mus)
    else
        Quaternion(change_basis_core(x,mus)...)
    end
end

function change_basis_core{T<:Quaternion}(x::AbstractArray{T}, mus::NTuple{3,T})
    x_im = imag(x)
    return real(x), -real(map(x->x|mus[1],x_im)), -real(map(x->x|mus[2],x_im)), -real(map(x->x|mus[3],x_im))
end

function change_basis_core{T<:Real, S<:Quaternion}(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, d::AbstractArray{T}, mus::NTuple{3,S})
    return a + b*mus[1] + c*mus[2] + d*mus[3]
end

function qconv_l(At,Bt,mus)
    mmus = (-mus[1], -mus[2], mus[3])
    Aa, Ab, Ac, Ad = change_basis_core(At, mus)
    Ba, Bb, Bc, Bd = change_basis_core(Bt, mus)

    CA1, CA2 = complex(Aa, Ab), complex(Ac, Ad)
    CB1, CB2 = complex(Ba, Bb), complex(Bc, Bd)

    p = plan_fft(CA1)
    fCA1, fCA2 = p*CA1, p*CA2
    fCB1, fCB2 = p*CB1, p*CB2

    fBp = change_basis_core(real(fCB1), imag(fCB1), real(fCB2), imag(fCB2), mus)
    fBm = change_basis_core(real(fCB1), imag(fCB1), real(fCB2), imag(fCB2), mmus)

    return ((real(fCA1)+imag(fCA1)*mus[1]) .* fBp) + (((real(fCA2)+imag(fCA2)*mus[1])*mus[2]) .* fBm)
end

function qconv_r(At,Bt,mus)
    mmus = (-mus[1], -mus[2], mus[3])
    Ba, Bb, Bc, Bd = change_basis_core(Bt, mus)
    Aa, Ab, Ac, Ad = change_basis_core(At, mus)

    CB1, CB2 = complex(Ba, Bb), complex(Bc, -Bd)
    CA1, CA2 = complex(Aa, Ab), complex(Ac, -Ad)

    p = plan_fft(CB1)
    fCB1, fCB2 = p*CB1, p*CB2
    fCA1, fCA2 = p*CA1, p*CA2

    fAp = change_basis_core(real(fCA1), imag(fCA1), real(fCA2), -imag(fCA2), mus)
    fAm = change_basis_core(real(fCA1), imag(fCA1), real(fCA2), -imag(fCA2), mmus)

    return (fAp .* (real(fCB1)+imag(fCB1)*mus[1])) + (fAm .* ((real(fCB2)-imag(fCB2)*mus[1])*mus[2]))
end

function qconv{T<:Quaternion}(A::AbstractArray{T}, B::AbstractArray{T}; mu::Union{T,NTuple{3,T}}=getbasis(eltype(A)), LR=:left)
    qconv_core = 
    if LR == :left
        qconv_l
    elseif LR == :right  
        qconv_r
    else
        error("LR must be :left or :right.")
    end

    mus = orthonormal_basis(mu...)

    sa, sb = size(A), size(B)
    w, h = sa[1]+sb[1]-1, sa[2]+sb[2]-1
    At = zeros(eltype(A), w, h)
    Bparat = zeros(eltype(A), w, h)
    Bperpt = zeros(eltype(A), w, h)

    At[1:sa[1], 1:sa[2]] = A
    Bparat[1:sb[1], 1:sb[2]] = B .* conj(B)
    Bperpt[1:sb[1], 1:sb[2]] = B .* B
    Bparat /= sum(map(norm, Bparat))
    Bperpt /= sum(map(norm, Bperpt))

    Apara = map(x->para(x, mus[1]), At)
    Aperp = map(x->perp(x, mus[1]), At)

    fApara = qconv_core(Apara, Bparat, mus)
    fAperp = qconv_core(Aperp, Bperpt, mus)

    return iqft(fApara + fAperp, mu=mus, LR=LR)
end

end # module
