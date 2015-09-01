using Quaternions
using QuaternionFourierTransform
using Images
using Colors
using TestImages

function saveimg{T<:Real}(x::AbstractArray{Quaternion{T}},fname::String)
    m = cat(3, imagi(x), imagj(x), imagk(x))
    m = normalize(m)
    saveimg(m,fname)
end

function normalize(m::AbstractArray)
    mmax = maximum(m)
    mmin = minimum(m)
    return (m - mmin)/(mmax-mmin)
end

function saveimg(x::AbstractArray,fname::String)
    img = convert(Image,x)
    img.properties["spatialorder"] = ["x","y"]
    imwrite(img,fname)
end


# load image
img = testimage("lena_color_256")


# create quaternion matrix from image data
r = float(red(img.data))
g = float(green(img.data))
b = float(blue(img.data))
rezero = zeros(size(img))
qmat = quaternion(rezero, r, g, b)

# execute qft
q_qft = qft(qmat)


# visualize the result
m = fftshift(q_qft)
mnorm = map(norm,m)
marg = map(m,mnorm) do x,y
    if y==zero(real(x))
        zero(real(x))
    else
        acos(real(x)/norm(x))/pi
    end
end
maxis = map(m,mnorm) do x,y
    if y==zero(typeof(y))
        quaternion(y)
    else
        imag(x)/y
    end
end

mmax = maximum(mnorm)
modulus = map(x->log(1+x)/log(1+mmax),mnorm) # log scaled
phases = map(x->HSV(x*360,1.0,1.0*sign(x)), marg)
axis = map(x->RGB(0.5*(imagi(x)+1),0.5*(imagj(x)+1),0.5*(imagk(x)+1)), maxis)

saveimg(modulus, "modulus.png")
saveimg(phases, "phases.png")
saveimg(axis, "axis.png")


# execute inverse qft
q_iqft = iqft(q_qft)
saveimg(q_iqft, "inv.png")


# high/low-pass filter example
q_high = copy(m)
q_high[end/8*3+1:end/8*5,end/8*3+1:end/8*5] = zero(eltype(q_high))

q_low = zeros(eltype(q_qft),size(q_qft))
q_low[end/8*3+1:end/8*5,end/8*3+1:end/8*5] = m[end/8*3+1:end/8*5,end/8*3+1:end/8*5]

q_high = fftshift(q_high)
q_low = fftshift(q_low)

q_iqft_high = iqft(q_high)
q_iqft_low = iqft(q_low)

saveimg(q_iqft_high, "high.png")
saveimg(q_iqft_low, "low.png")

