using Quaternions
using QuaternionFourierTransform
using Images
using Colors
using TestImages
using FFTW
using LinearAlgebra

function qmat2img(x::AbstractArray{T}) where T<:Quaternion
    m = cat(imagi(x), imagj(x), imagk(x), dims=3)
    m = permutedims(m, (3,1,2))
    return colorview(RGB, m)
end

function normalize(m::AbstractArray)
    mmax = maximum(m)
    mmin = minimum(m)
    return mmax==mmin ? zeros(size(m)) : (m - mmin)/(mmax-mmin)
end

function img2qmat(img::Array)
    channels = float(channelview(img))
    @assert size(channels)[1] == 3
    rezero = zeros(size(img))
    return quaternion(rezero, channels[1,:,:], channels[2,:,:], channels[3,:,:])
end

function calc_arg(qfreq::AbstractArray{T},qnorm::AbstractArray{S}) where {T<:Quaternion,S<:Real}
    map(qfreq,qnorm) do x,y
        if iszero(y)
            zero(S)
        else
            acos(real(x)/norm(x))
        end
    end
end

function calc_axis(qfreq::AbstractArray{T},qnorm::AbstractArray{S}) where {T<:Quaternion,S<:Real}
    map(qfreq,qnorm) do x,y
        if iszero(y)
            quaternion(y)
        else
            imag(x)/y
        end
    end
end

function test_qft(qimg::AbstractArray{T}) where T<:Quaternion
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
    save("modulus.png", colorview(Gray,modulus))
    save("phase.png", phase)
    save("axis.png", axis)

    # exec inverse quaternion fft
    qinv = iqft(qfreq)
    save("inv.png", qmat2img(qinv))
end

function test_freqfilter(qimg::AbstractArray{T}) where T<:Quaternion
    # exec quaternion fft
    qfreq = qft(qimg)

    # swap quadrants
    qfreqs = fftshift(qfreq)

    # high-pass filter
    wrange = (size(qfreqs,1)>>3)*3+1:(size(qfreqs,1)>>3)*5
    hrange = (size(qfreqs,2)>>3)*3+1:(size(qfreqs,2)>>3)*5
    q_high = copy(qfreqs)
    q_high[wrange, hrange] .= zero(T)

    # low-pass filter
    q_low = zeros(T,size(qfreq))
    q_low[wrange, hrange] = qfreqs[wrange, hrange]

    # exec inverse quaternion fft
    q_high = fftshift(q_high)
    q_low = fftshift(q_low)
    q_iqft_high = iqft(q_high)
    q_iqft_low = iqft(q_low)

    # save as image
    save("high.png", qmat2img(q_iqft_high))
    save("low.png", qmat2img(q_iqft_low))
end

function test_convolution(qimg::AbstractArray{T}) where T<:Quaternion
    # gray line
    mu = QuaternionFourierTransform.defaultbasis

    # create filter (Prewitt inspired color chromatic edge detection)
    qfilter = zeros(Quaternion{Float64}, 3,3)
    q = exp(mu*pi/4)
    qfilter[1,1:end] .= q
    qfilter[end,1:end] .= conj(q)
    qfilter /= sum(map(norm,qfilter))

    # apply filter (left and right)
    qc = qconv(qimg, qfilter, LR=:left)

    # save as image
    qperp = map(x->perp(x,mu), qc)
    qpara = map(x->para(x,mu), qc)
    save("conv_perp.png", qmat2img(qperp))
    save("conv_para.png", qmat2img(qpara))
    save("conv.png", qmat2img(qc))
    save("filter.png", qmat2img(qfilter))
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
