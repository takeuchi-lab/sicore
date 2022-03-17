module SelectiveInference
export SelectiveInferenceNormSE, addSelectionEvent, test

using LinearAlgebra
using Distributions


mutable struct SelectiveInferenceNormSE
    #= 
    Base inference class for a test statistic which follows normal
    distribution under null.

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
    c::Vector{Float64}
    z::Vector{Float64}
    lower::Float64
    upper::Float64
    concaveIntervals::Vector{Any}
    summary::Dict{String,Int}
    
    function SelectiveInferenceNormSE(data::Union{Vector{Float64},Matrix{Float64}}, var::Union{Float64,Matrix{Float64}}, eta::Vector{Float64}, initLower = -Inf, initUpper = Inf)
        data::Vector{Float64} = isa(data, Matrix{Float64}) ? vec(data) : data
        len::Int = length(data)
        cov::Matrix{Float64} = isa(var, Union{Float64,Int}) ? var * Matrix{Int}(I, len, len) : var
        stat::Float64 = eta' * data
        sigmaEta::Vector{Float64} = cov * eta
        etaSigmaEta::Float64 = eta' * sigmaEta
        c::Vector{Float64} = sigmaEta / etaSigmaEta
        z::Vector{Float64} = data - stat * c
        lower::Float64 = initLower
        upper::Float64 = initUpper
        concaveIntervals::Vector{Tuple{Float64,Float64}} = []
        summary = Dict([("linear", 0), ("convex", 0), ("concave", 0)])
        
        new(data, var, eta, len, cov, stat, sigmaEta, etaSigmaEta, c, z, lower, upper, concaveIntervals, summary)
    end
end

function tnCdf(x, interval::Matrix{Float64})::Float64
    #= 
    CDF of a truncated normal distribution.

    Args:
        x (float): Return the value at `x`.
        interval (Matrix): Truncation interval [L1 U1; L2 U2; ...]
    Returns:
        float: CDF value at `x`. =#
    num = denom = 0
    insideFlag = false
    
    for i in 1:size(interval)[1]
        lower = interval[i, 1]
        upper = interval[i, 2]
        diff = cdf(Normal(0, 1), BigFloat(upper)) - cdf(Normal(0, 1), BigFloat(lower))
        denom += diff
        if lower <= x <= upper
            num += cdf(Normal(0, 1), BigFloat(x)) - cdf(Normal(0, 1), BigFloat(lower))
            insideFlag = true
        elseif upper < x
            num += diff
        end
    end
    
    if !insideFlag
        error("Value x is outside the interval.")
    end
            
    num / denom
end

function addSelectionEvent(self::SelectiveInferenceNormSE, A::Union{Matrix{Float64},Nothing} = nothing, b::Union{Vector{Float64},Nothing} = nothing, c::Union{Float64,Nothing} = nothing)
    #= 
        Add a selection event {x'Ax+b'x+c≦0}.

        Args:
            A (Matrix, optional): `nd`*`nd` matrix. Set None if `A` is unused. Defaults to None.
            b (Vector, optional): `nd` dimensional vector. Set None if `b` is unused. Defaults to None.
            c (float, optional): Constant. Set None if `c` is unused. Defaults to None. =#
    
    α = β = γ = 0
    
    if !isnothing(A)
        cA = vec(Matrix(self.c' * A))
        zA = vec(Matrix(self.z' * A))
        α += cA' * self.c
        β += zA' * self.c + cA' * self.z
        γ += zA' * self.z
    end
    if !isnothing(b)
        β += b' * self.c
        γ += b' * self.z
    end
    if !isnothing(c)
        γ += c
    end
    
    cutInterval(self, α, β, γ)
end


function cutInterval(self::SelectiveInferenceNormSE, a::Float64, b::Float64, c::Float64, tau = false)
    #= 
        Truncate the interval with a quadratic inequality `ατ^2+βτ+γ≦0`,
        where `τ` is test statistic, and `β`, `γ` are function of `c`, `z`
        respectively. We can also use a auadratic inequality `ατ^2+κτ+λ≦0`,
        where `κ`, `λ` are function of `c`, `x`. This method truncates the
        interval only when the inequality is convex in order to reduce
        calculation cost. Truncation intervals for concave inequalities are
        stored in ``self.__concave_intervals``. The final truncation intervals
        are calculated when ``self.get_intervals()`` is called.

        Args:
            a (float): `α`. Set 0 if the inequality is linear.
            b (float): `β` or `κ`.
            c (float): `γ` or `λ`.
            tau (bool, optional): Set False when the inputs are `β` and `γ`, and
                True when they are `κ` and `λ`. Defaults to False.

        Raises:
            ValueError: If the test direction of interest does not intersect
                with the inequality or the polytope. =#
    tau = tau ? self.stat : 0
    
    threshold = 1e-10
    if -threshold < a < threshold
        a = 0
    end
    if -threshold < b < threshold
        b = 0
    end
    if -threshold < c < threshold
        c = 0
    end
    
    if iszero(a)
        if iszero(b)
            if c <= 0
                return
            else
                error("Test direction of interest does not intersect with the inequality.")
            end
        elseif b < 0
            self.lower = max(self.lower, -c / b - tau)
        else
            self.upper = min(self.upper, -c / b - tau)
        end
        self.summary["linear"] += 1
    elseif a > 0
        disc = b^2 - 4 * a * c
#         if -threshold < disc < threshold
#             disc = 0
#         end
        disc <= 0 && error(disc, ": Test direction of interest does not intersect with the inequality.")
        
        self.lower = max(self.lower, (-b - sqrt(disc)) / (2 * a) + tau)
        self.upper = min(self.upper, (-b + sqrt(disc)) / (2 * a) + tau)
        self.summary["convex"] += 1
    else
        disc = b^2 - 4 * a * c
#         if -threshold < disc < threshold
#             disc = 0
#         end
        disc <= 0 && return
        
        lower = (-b + sqrt(disc)) / (2 * a) + tau
        upper = (-b - sqrt(disc)) / (2 * a) + tau
        push!(self.concaveIntervals, (lower, upper))
        self.summary["concave"] += 1
    end
        
    self.lower >= self.upper && error("Test direction of interest does not intersect with the polytope.")
end


function getInterval(self::SelectiveInferenceNormSE)::Matrix{Float64}
    #= 
        Get truncation intervals.

        Returns:
            list: List of truncation intervals [L1 U1; L2 U2; ...] =#
    intervals = [self.lower self.upper]
    for (lower, upper) in self.concaveIntervals
        if lower <= intervals[1, 1] < upper < intervals[end, 2]
            # truncate the left side of the intervals
            for i in 1:size(intervals)[1]
                if upper <= intervals[i, 1]
                    intervals = intervals[i:end, :]
                    break
                elseif intervals[i, 1] < upper < intervals[i, 2]
                    intervals = intervals[i:end, :]
                    intervals[1, 1] = upper
                    break
                end
            end
        elseif intervals[1, 1] < lower < intervals[end, 2] <= upper
            # truncate the right side of the intervals
            for i in 1:size(intervals)[1]
                idx = size(intervals)[1] + 1 - i
                if intervals[idx, 2] <= lower
                    intervals = intervals[1:idx, :]
                    break
                elseif intervals[idx, 1] < lower < intervals[idx, 2]
                    intervals = intervals[1:idx, :]
                    intervals[end, 2] = lower
                    break
                end
            end
        elseif intervals[1, 1] < lower && upper < intervals[end, 2]
            # truncate the middle part of the intervals
            leftIntervals = Matrix{Float64}(undef, 1, 2)
            rightIntervals = Matrix{Float64}(undef, 1, 2)
            for i in 1:size(intervals)[1]
                if upper <= intervals[i, 1]
                    rightIntervals = intervals[i:end, :]
                    break
                elseif intervals[i, 1] < upper < intervals[i, 2]
                    rightIntervals  = vcat([upper intervals[i, 2]], intervals[i + 1:end, :])
                    break
                end
            end
            for i in 1:size(intervals)[1]
                idx = size(intervals)[1] + 1 - i
                if intervals[idx, 2] <= lower
                    leftIntervals = intervals[1:idx, :]
                    break
                elseif intervals[idx, 1] < lower < intervals[idx, 2]
                    leftIntervals = vcat(intervals[1:idx - 1, :], [intervals[idx, 1] lower])
                    break
                end
            end
            intervals = vcat(leftIntervals, rightIntervals)
        elseif lower <= intervals[1, 1] && intervals[end, 2] <= upper
            error("Test direction of interest does not intersect with the polytope.")
        end
    end
    
    intervals
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


function standardize(x::Union{Float64,Vector{Float64},Matrix{Float64}}, mean = 0, var = 1)
    sd = sqrt(var)
    (x .- mean) / sd
end


function test(self::SelectiveInferenceNormSE, tail = "double", popmean = 0)
    #= 
        Perform selective statistical testing.

        Args:
            tail (str, optional): 'double' for double-tailed test, 'right' for
                right-tailed test, and 'left' for left-tailed test. Defaults to
                'double'.
            popmean (float, optional): Population mean of `η'x` under null
                hypothesis is true. Defaults to 0.

        Returns:
            float: p-value =#
    interval = getInterval(self)
    
    stat = standardize(self.stat, popmean, self.etaSigmaEta)
    normInterval = standardize(interval, popmean, self.etaSigmaEta)
    F = tnCdf(stat, normInterval)
    calcPvalue(F)
end

end