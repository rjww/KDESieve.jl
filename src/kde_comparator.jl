struct KDEComparator{T₁ <: KernelDensity.UnivariateKDE,
                     T₂ <: KernelDensity.UnivariateKDE}
    pdf₁::KernelDensity.InterpKDE{T₁}
    pdf₂::KernelDensity.InterpKDE{T₂}

    function KDEComparator(samples::T₁,
                           targets::T₂,
                           weights::T₃) where {T₁ <: AbstractMatrix,
                                               T₂ <: AbstractVector,
                                               T₃ <: AbstractVector}
        X = samples
        t = targets
        w = weights
        h = vec(w' * X)

        # Separate the random projections by label.
        # TODO Perhaps make more robust by deriving (binary) values from t,
        # rather than assuming they'll always be 1 and not-1.
        h₀ = h[t .== 1]
        h₁ = h[t .!= 1]

        # Set lower and upper bounds on the bandwidth of the KDEs, with some
        # allowance for interpolating values that will fall outside the range of
        # the training data.
        # TODO The constant term here should be a parameter to the model.
        bounds = (min(minimum(h₀), minimum(h₁)) - 1,
                  max(maximum(h₀), maximum(h₁)) + 1)

        # Calculate kernel density estimates for each per-label subset.
        # TODO The npoints value here should be a parameter to the model.
        kde₀ = KernelDensity.kde_lscv(h₀, npoints = 100, boundary = bounds)
        kde₁ = KernelDensity.kde_lscv(h₁, npoints = 100, boundary = bounds)

        # Interpret each KDE as a probability density function.
        pdf₀ = KernelDensity.InterpKDE(kde₀)
        pdf₁ = KernelDensity.InterpKDE(kde₁)

        param(::KernelDensity.InterpKDE{T}) where {T} = T
        new{param(pdf₀),param(pdf₁)}(pdf₀, pdf₁)
    end
end

(kde::KDEComparator)(x::T) where {T <: Number} = (Distributions.pdf(kde.pdf₁, x) -
                                                  Distributions.pdf(kde.pdf₂, x))
