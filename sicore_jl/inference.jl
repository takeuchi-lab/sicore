module Inference
export InferenceNorm, naiveTest

using LinearAlgebra
using Distributions


struct InferenceNorm
    #= 
    Base inference class for a test statistic which follows normal distribution under null.
    Args:
        data (Vector): Observation data of length `nd`.
        var (float, Matrix): Value of known variance, or `nd` * `nd` covariance matrix.
        eta (Vector): `nd` dimensional vector of test direction. =#
    
    data::Vector{Float64}
    var::Union{Float64,Matrix{Float64}}
    eta::Vector{Float64}
    len::Int
    cov::Matrix{Float64}
    stat::Float64
    sigmaEta::Vector{Float64}
    etaSigmaEta::Float64
    
    function InferenceNorm(data::Union{Vector{Float64},Matrix{Float64}}, var::Union{Float64,Matrix{Float64}}, eta::Vector{Float64})
        data::Vector{Float64} = isa(data, Matrix{Float64}) ? vec(data) : data
        len::Int = length(data)
        cov::Matrix{Float64} = isa(var, Union{Float64,Int}) ? var * Matrix{Int}(I, len, len) : var
        stat::Float64 = eta' * data
        sigmaEta::Vector{Float64} = cov * eta
        etaSigmaEta::Float64 = eta' * sigmaEta
        
        new(data, var, eta, len, cov, stat, sigmaEta, etaSigmaEta)
    end
end

function calcPvalue(F, tail = "double")
    #= 
    Calculate p-value.
    Args:
        F (float): CDF value at the observed test statistic.
        tail (str, optional): Set 'double' for double-tailed test, 'right' for
            right-tailed test, and 'left' for left-tailed test. Defaults to
            'double'.
    Returns:
        float: p-value =#
    if tail == "double"
        return 2 * min(F, 1 - F)
    elseif tail == "right"
        return 1 - F
    elseif tail == "left"
        return F
    end
end


function standardize(x::Union{Float64,Vector{Float64}}, mean = 0, var = 1)
    sd = sqrt(var)
    (x .- mean) / sd
end


function naiveTest(inferenceNorm::InferenceNorm, tail = "double", popmean = 0)
    #= 
    Perform naive statistical testing.
        Args:
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to
                'double'.
            popmean (float, optional): Population mean of `η'x` under null
                hypothesis is true. Defaults to 0.
        Returns:
            float: p-value =#
    stat = standardize(inferenceNorm.stat, popmean, inferenceNorm.etaSigmaEta)
    F = cdf.(Normal(0, 1), stat)[1]
    return calcPvalue(F), stat
end

end