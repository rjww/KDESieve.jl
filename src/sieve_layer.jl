struct SieveLayer{T₁ <: Number}
    weights::Matrix{T₁}
    comparators::Vector{Comparator}
    next::Union{SieveLayer{T₁},Nothing}

    function SieveLayer{T₁}(samples::T₂,
                            targets::T₃,
                            n_projections::Int,
                            remaining_depth::Int,
                            min_split::Int) where {T₁ <: Number,
                                                   T₂ <: AbstractMatrix,
                                                   T₃ <: AbstractVector}
        X = samples
        t = targets
        L = n_projections
        D, N = size(X)

        # Initialize the random weights to the layer W, then instantiate a
        # Comparator for each random projection vector, a row in W.
        W = gaussian_projection_matrix(T₁, L, D)
        comparators = [Comparator(X, t, W[l,:]) for l in 1:L]

        # Calculate the set of random projections H of X onto W, then apply the
        # corresponding comparator across each set of projected values, giving a
        # set of predictions Y.
        H = W * X
        Y = zeros(T₁, L, N)
        for l in 1:L
            f = comparators[l]
            Y[l,:] .= f.(H[l,:])
        end

        # Get the indices of all samples for which all comparators aren't in
        # consensus, and separate out the samples and targets on those indices.
        indices = [!consensus(Y[:,n]) for n in 1:N]
        X₀ = X[:,indices]
        t₀ = t[indices]

        # TODO Don't go any deeper if the max depth has been reached, if the
        # number of ambiguously predicted samples is less than the minimum split
        # value, or if all of these samples correspond to one label. (Is this
        # how that last case should be handled?)
        if remaining_depth == 0 || length(t₀) < min_split ||
                all(t₀ .== 1) || all(t₀ .!= 1)
            next = nothing
        else
            next = SieveLayer{T₁}(X₀, t₀, L, remaining_depth - 1, min_split)
        end

        new{T₁}(W, comparators, next)
    end
end

function gaussian_projection_matrix(::Type{T},
                                    n_projections::Int,
                                    n_features::Int) where {T <: Number}
    randn(T, n_projections, n_features) ./ sqrt(n_features)
end

function consensus(sample::T₁) where {T₁ <: AbstractVector}
    signs = sign.(sample)
    all(signs .== 1) || all(signs .!= 1)
end
