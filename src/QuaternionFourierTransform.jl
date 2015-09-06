isdefined(Base, :__precompile__) && __precompile__()

module QuaternionFourierTransform

using Quaternions
export qft, iqft, qconv2
const defaultbasis = quaternion(0,1,1,1)/sqrt(3)

function qft{T<:Quaternion}(x::AbstractArray{T}; mu::Union(T,NTuple{3,T})=defaultbasis, LR::Symbol=:left)
    mus = orthonormal_basis(mu...)
    if LR == :left
        qft_core(x, fft, mus, 1)
    elseif LR == :right
        qft_core(x, fft, mus, -1)
    else
        error("LR must be :left or :right.")
    end
end

function iqft{T<:Quaternion}(x::AbstractArray{T}; mu::Union(T,NTuple{3,T})=defaultbasis, LR::Symbol=:left)
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
    
    if maximum(m*m'-eye(3))>10*eps() 
        warn("The basis matrix is not accurately orthogonal.")
    end

    return (m1,m2,m3)
end

function change_basis{T<:Quaternion}(x::AbstractArray{T}, mus::NTuple{3,T}, inverse::Bool=false)
    if inverse
        change_basis_core(real(x),imagi(x),imagj(x),imagk(x),mus)
    else
        quaternion(change_basis_core(x,mus)...)
    end
end

function change_basis_core{T<:Quaternion}(x::AbstractArray{T}, mus::NTuple{3,T})
    x_im = imag(x)
    return real(x), -real(map(x->x|mus[1],x_im)), -real(map(x->x|mus[2],x_im)), -real(map(x->x|mus[3],x_im))
end

function change_basis_core{T<:Real, S<:Quaternion}(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, d::AbstractArray{T}, mus::NTuple{3,S})
    return a + b*mus[1] + c*mus[2] + d*mus[3]
end

function qconv2{T<:Quaternion}(A::AbstractArray{T}, B::AbstractArray{T}; mu::Union(T,NTuple{3,T})=defaultbasis, LR::Symbol=:left)
    mus = orthonormal_basis(mu...)
    sa, sb = size(A), size(B)
    At = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
    Bt = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
    At[1:sa[1], 1:sa[2]] = A
    Bt[1:sb[1], 1:sb[2]] = B

    fC = if LR == :left
        qconv2_l(At,Bt,mus)
    elseif LR == :right
        qconv2_r(At,Bt,mus)
    else
        error("LR must be :left or :right.")
    end
    return iqft(fC,mu=mus,LR=LR)
end

function qconv2_l(At,Bt,mus)
    mmus = (-mus[1],-mus[2],mus[3])
    Aa, Ab, Ac, Ad = change_basis_core(At, mus)
    Bpa, Bpb, Bpc, Bpd = change_basis_core(Bt, mus)

    CA1, CA2 = complex(Aa, Ab), complex(Ac, Ad)
    CBp1, CBp2 = complex(Bpa, Bpb), complex(Bpc, Bpd)
    CBm1, CBm2 = complex(Bpa, -Bpb), complex(-Bpc, Bpd)

    p = plan_fft(CA1)
    fCA1, fCA2 = p(CA1), p(CA2)
    fCBp1, fCBp2 = p(CBp1), p(CBp2)
    fCBm1, fCBm2 = p(CBm1), p(CBm2)

    fBp = change_basis_core(real(fCBp1), imag(fCBp1), real(fCBp2), imag(fCBp2), mus)
    fBm = change_basis_core(real(fCBm1), imag(fCBm1), real(fCBm2), imag(fCBm2), mmus)

    return ((real(fCA1)+imag(fCA1)*mus[1]) .* fBp) + (((real(fCA2)+imag(fCA2)*mus[1])*mus[2]) .* fBm)
end

function qconv2_r(At,Bt,mus)
    mmus = (-mus[1],-mus[2],mus[3])
    Ba, Bb, Bc, Bd = change_basis_core(Bt, mus)
    Apa, Apb, Apc, Apd = change_basis_core(At, mus)

    CB1, CB2 = complex(Ba, Bb), complex(Bc, -Bd)
    CAp1, CAp2 = complex(Apa, Apb), complex(Apc, -Apd)
    CAm1, CAm2 = complex(Apa, -Apb), complex(-Apc, -Apd)

    p = plan_fft(CB1)
    fCB1, fCB2 = p(CB1), p(CB2)
    fCAp1, fCAp2 = p(CAp1), p(CAp2)
    fCAm1, fCAm2 = p(CAm1), p(CAm2)

    fAp = change_basis_core(real(fCAp1), imag(fCAp1), real(fCAp2), -imag(fCAp2), mus)
    fAm = change_basis_core(real(fCAm1), imag(fCAm1), real(fCAm2), -imag(fCAm2), mmus)

    return (fAp .* (real(fCB1)+imag(fCB1)*mus[1])) + (fAm .* ((real(fCB2)-imag(fCB2)*mus[1])*mus[2]))
end

function qconv2{T<:Quaternion}(A::AbstractArray{T}, Bl::AbstractArray{T}, Br::AbstractArray{T}; mu::Union(T,NTuple{3,T})=defaultbasis)
    mus = orthonormal_basis(mu...)

    sa, sb = size(A), size(Bl)
    w, h = sa[1]+sb[1]-1, sa[2]+sb[2]-1
    At = zeros(eltype(A), w, h)
    Bparat = zeros(eltype(A), w, h)
    Bperpt = zeros(eltype(A), w, h)

    At[1:sa[1], 1:sa[2]] = A
    Bparat[1:sb[1], 1:sb[2]] = Bl .* Br
    Bperpt[1:sb[1], 1:sb[2]] = Bl .* conj(Br)

    Apara = map(x->para(x,mus[1]),At)
    Aperp = map(x->perp(x,mus[1]),At)

    fApara = qconv2_l(Apara, Bparat, mus)
    fAperp = qconv2_l(Aperp, Bperpt, mus)

    return iqft(fApara + fAperp,mu=mus)
end

end # module
