using QuaternionFourierTransform
using Quaternions
using Test

x = rand(QuaternionF64, 100, 100)
@test isapprox(iqft(qft(x, LR=:left), LR=:left), x)
@test isapprox(iqft(qft(x, LR=:right), LR=:right), x)

f = rand(QuaternionF64, 3, 3)
@test qconv(x, f, LR=:left) isa typeof(x)
@test qconv(x, f, LR=:right) isa typeof(x)
