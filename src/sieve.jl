struct Sieve{T₁ <: Number}
    n_projections::Int
    n_layers::Int
    head::SieveLayer{T₁}

    function Sieve{T₁}(samples::T₂,
                          targets::T₃,
                          n_projections::Int;
                          max_layers::Int = 3,
                          min_split::Int = 2) where {T₁ <: Number,
                                                     T₂ <: AbstractMatrix,
                                                     T₃ <: AbstractVector}
        X = samples
        t = targets
        L = n_projections
        head = SieveLayer{T₁}(X, t, L, max_layers - 1, min_split)

        n_layers = 0
        current = head
        while !isnothing(current)
            n_layers += 1
            current = current.next
        end

        new{T₁}(n_projections, n_layers, head)
    end
end

function predict(sieve::T₁, samples::T₂) where {T₁ <: Sieve,
                                                T₂ <: AbstractMatrix}
    param(::Sieve{T}) where {T} = T

    X = samples
    L = sieve.n_projections
    N = last(size(X))
    Y = zeros(param(sieve), L * sieve.n_layers, N)

    layer = sieve.head
    i = 0

    while !isnothing(layer)
        W = layer.weights
        H = W * X
        comparators = layer.comparators

        for l in 1:L
            f = comparators[l]
            Y[i+l,:] .= f.(H[l,:])
        end

        layer = layer.next
        i += L
    end

    Y
end
