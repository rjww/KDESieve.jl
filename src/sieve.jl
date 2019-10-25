struct Layer{T <: Number}
    input_weights::Matrix{T}
    comparators::Vector{Comparator}
end

struct Sieve{T <: Number}
    n_projections::Int
    layers::Vector{Layer{T}}
    output_weights::Matrix{T}
    consensus_threshold::Real

    function Sieve{T₁}(samples::T₂,
                       targets::T₃,
                       n_projections::Int;
                       consensus_threshold::Real = 0.9,
                       max_layers::Int = 3,
                       min_split::Int = 2) where {T₁ <: Number,
                                                  T₂ <: AbstractMatrix,
                                                  T₃ <: AbstractVector}
        X = samples
        t = targets
        L = n_projections
        N = last(size(X))
        H = zeros(T₁, L * max_layers, N)
        layers = Vector{Layer{T₁}}()

        X₀ = X
        active_samples = Set{Int}(1:N)
        indices = get_indices(active_samples, N)

        for depth in 1:max_layers
            if length(active_samples) < min_split break end

            i₀ = (depth-1) * L + 1
            i₁ = i₀ + L - 1

            H₀ = @view H[i₀:i₁,indices]
            t₀ = @view t[indices]
            D, N₀ = size(X₀)

            W = gaussian_projection_matrix(T₁, L, D)
            fs = [Comparator(X₀, t₀, W[l,:]) for l in 1:L]
            H₀ .= project(X₀, W, fs)
            push!(layers, Layer{T₁}(W, fs))

            to_remove = Set{Int}()
            for n in active_samples
                if consensus(H[i₀:i₁,n], consensus_threshold)
                    push!(to_remove, n)
                end
            end
            setdiff!(active_samples, to_remove)
            indices = get_indices(active_samples, N)

            X₀ = @view H[i₀:i₁,indices]
        end

        ψ₀ = sum(t .!= 1) / N
        ψ₁ = sum(t .== 1) / N
        Ψ = LinearAlgebra.Diagonal([q != 1 ? ψ₀ : ψ₁ for q in t])
        H = [H; ones(eltype(H), 1, N)] * Ψ
        T = reshape(t, 1, :) * Ψ

        β = (T * H') * LinearAlgebra.pinv(H * H')

        new{T₁}(L, layers, β, consensus_threshold)
    end
end

function predict(sieve::T₁,
                 samples::T₂) where {T₁ <: Sieve,
                                     T₂ <: AbstractMatrix}
    param(::Sieve{T}) where {T} = T

    X = samples
    L = sieve.n_projections
    N = last(size(X))
    H = zeros(param(sieve), L * length(sieve.layers), N)

    X₀ = X
    active_samples = Set{Int}(1:N)
    indices = get_indices(active_samples, N)

    for (depth, layer) in enumerate(sieve.layers)
        # TODO Break on min split value.

        i₀ = (depth-1) * L + 1
        i₁ = i₀ + L - 1

        H₀ = @view H[i₀:i₁,indices]
        D, N₀ = size(X₀)

        W = layer.input_weights
        fs = layer.comparators
        H₀ .= project(X₀, W, fs)

        to_remove = Set{Int}()
        for n in active_samples
            if consensus(H[i₀:i₁,n], sieve.consensus_threshold)
                push!(to_remove, n)
            end
        end
        setdiff!(active_samples, to_remove)
        indices = get_indices(active_samples, N)

        X₀ = @view H[i₀:i₁,indices]
    end

    H = [H; ones(eltype(H), 1, N)]
    β = sieve.output_weights
    Y = β * H
    vec(Y)
end

function predict(sieve::T₁,
                 sample::T₂) where {T₁ <: Sieve,
                                    T₂ <: AbstractVector}
    y = predict(sieve, reshape(sample, :, 1))
    first(y)
end

function project(samples::T₁,
                 weights::T₂,
                 comparators::Vector{T₃}) where {T₁ <: AbstractMatrix,
                                                 T₂ <: AbstractMatrix,
                                                 T₃ <: Comparator}
    X = samples
    W = weights
    fs = comparators
    L = length(fs)
    D, N = size(X)

    H = W * X
    for l in 1:L
        H[l,:] .= fs[l].(H[l,:])
    end

    H
end

function gaussian_projection_matrix(::Type{T},
                                    n_projections::Int,
                                    n_features::Int) where {T <: Number}
    randn(T, n_projections, n_features) ./ sqrt(n_features)
end

function consensus(samples::T₁,
                   consensus_threshold::Real) where {T₁ <: AbstractVector}
    y = sign.(samples)
    th = consensus_threshold
    f(x) = x == 0 || x == 1
    g(x) = x == 0 || x == -1

    all(f.(y)) || all(g.(y))
end

function get_indices(active_samples::Set{Int}, N::Int)
    [n in active_samples for n in 1:N]
end
