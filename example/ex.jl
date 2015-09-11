using Quaternions
using QuaternionFourierTransform
using Images
using Colors
using TestImages

function saveimg{T<:Quaternion}(x::AbstractArray{T},fname::String)
    m = cat(3, imagi(x), imagj(x), imagk(x))
    m = normalize(m)
    saveimg(m,fname)
end

function normalize(m::AbstractArray)
    mmax = maximum(m)
    mmin = minimum(m)
    return mmax==mmin ? zeros(size(m)) : (m - mmin)/(mmax-mmin)
end

function saveimg(x::AbstractArray,fname::String)
    img = convert(Image,x)
    img.properties["spatialorder"] = ["x","y"]
    imwrite(img,fname)
end

function img2qmat(img::Image)
    r = float(red(img.data))
    g = float(green(img.data))
    b = float(blue(img.data))
    rezero = zeros(size(img))
    return quaternion(rezero, r, g, b)
end

function calc_arg{T<:Quaternion,S<:Real}(qfreq::AbstractArray{T},qnorm::AbstractArray{S})
    map(qfreq,qnorm) do x,y
        if y==zero(S)
            zero(S)
        else
            acos(real(x)/norm(x))
        end
    end
end

function calc_axis{T<:Quaternion,S<:Real}(qfreq::AbstractArray{T},qnorm::AbstractArray{S})
    map(qfreq,qnorm) do x,y
        if y==zero(S)
            quaternion(y)
        else
            imag(x)/y
        end
    end
end

function test_qft{T<:Quaternion}(qimg::AbstractArray{T})
    # exec quaternion fft
    qfreq = qft(qimg)

    # swap quadrants
    qfreqs = fftshift(qfreq)

    # calc values
    qnorm = map(norm, qfreqs)
    qarg = calc_arg(qfreqs, qnorm)
    qaxis = calc_axis(qfreqs, qnorm)

    # visualize modulus, phase and axis
    mmax = maximum(qnorm)
    modulus = map(x->log(1+x)/log(1+mmax), qnorm) # log scaled
    phase = map(x->HSV(x/pi*360, 1.0, 1.0*sign(x)), qarg)
    axis = map(x->RGB(0.5*(imagi(x)+1), 0.5*(imagj(x)+1), 0.5*(imagk(x)+1)), qaxis)

    # save as images
    saveimg(modulus, "modulus.png")
    saveimg(phase, "phase.png")
    saveimg(axis, "axis.png")

    # exec inverse quaternion fft
    qinv = iqft(qfreq)
    saveimg(qinv, "inv.png")
end

function test_freqfilter{T<:Quaternion}(qimg::AbstractArray{T})
    # exec quaternion fft
    qfreq = qft(qimg)

    # swap quadrants
    qfreqs = fftshift(qfreq)

    # high-pass filter
    q_high = copy(qfreqs)
    q_high[end/8*3+1:end/8*5,end/8*3+1:end/8*5] = zero(T)

    # low-pass filter
    q_low = zeros(T,size(qfreq))
    q_low[end/8*3+1:end/8*5,end/8*3+1:end/8*5] = qfreqs[end/8*3+1:end/8*5,end/8*3+1:end/8*5]

    # exec inverse quaternion fft
    q_high = fftshift(q_high)
    q_low = fftshift(q_low)
    q_iqft_high = iqft(q_high)
    q_iqft_low = iqft(q_low)

    # save as image
    saveimg(q_iqft_high, "high.png")
    saveimg(q_iqft_low, "low.png")
end

function test_convolution{T<:Quaternion}(qimg::AbstractArray{T})
    # gray line
    mu = QuaternionFourierTransform.defaultbasis

    # create filter (Prewitt inspired color chromatic edge detection)
    qfilter = zeros(Quaternion{Float64}, 3,3)
    q = exp(mu*pi/4)
    qfilter[1,1:end] = q
    qfilter[end,1:end] = conj(q)
    qfilter /= sum(map(norm,qfilter))

    # apply filter (left and right)
    qc = qconv(qimg, qfilter, LR=:left)

    # save as image
    qperp = map(x->perp(x,mu), qc)
    qpara = map(x->para(x,mu), qc)
    saveimg(qperp, "conv_perp.png")
    saveimg(qpara, "conv_para.png")
    saveimg(qc, "conv.png")
    saveimg(qfilter, "filter.png")
end

function main()
    # load test image
    img = testimage("lena_color_256")

    # convert color image to quaternion matrix
    qimg = img2qmat(img)

    # exec quaternion fft examples
    test_qft(qimg)
    test_freqfilter(qimg)
    test_convolution(qimg)
end

main()

