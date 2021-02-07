struct Dataset{T<:Real}

    X::AbstractMatrix{T}
    y::AbstractVector{T}
    n::Int
    nfeatures::Int
    weighted::Bool
    weights::Union{AbstractVector{T}, Nothing}
    varMap::Array{String, 1}

end

function Dataset(
        X::AbstractMatrix{T},
        y::AbstractVector{T};
        weights::Union{AbstractVector{T}, Nothing}=nothing,
        varMap::Union{Array{String, 1}, Nothing}=nothing
       ) where {T<:Real}

    n = size(X, 2)
    nfeatures = size(X, 1)
    weighted = weights !== nothing
    if varMap == nothing
        varMap = ["x$(i)" for i=1:nfeatures]
    end

    if (typeof(X) <: CuArray) || (typeof(y) <: CuArray) || (weighted == true && typeof(weights) <: CuArray)
        if !((typeof(X) <: CuArray) && (typeof(y) <: CuArray) && (weighted == false || typeof(weights) <: CuArray))
            AssertionError("If one array is on the GPU, they all need to be on the GPU.")
        end
    end

    return Dataset{T}(X, y, n, nfeatures, weighted, weights, varMap)

end

