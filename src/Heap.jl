using CUDA
using SparseArrays
using FromFile
@from "Core.jl" import Node, CONST_TYPE, Options
@from "EquationUtils.jl" import countDepth

mutable struct EquationHeap
    operator::Array{Int, 1}
    constant::Array{CONST_TYPE, 1}
    feature::Array{Int, 1} #0=>use constant
    degree::Array{Int, 1} #0 for degree => stop the tree!

    EquationHeap(n) = new(zeros(Int, n), zeros(CONST_TYPE, n), zeros(Int, n), zeros(Int, n))
end

mutable struct EquationHeaps
    operator::Array{Int, 2}
    constant::Array{CONST_TYPE, 2}
    feature::Array{Int, 2} #0=>use constant
    degree::Array{Int, 2} #0 for degree => stop the tree!

    EquationHeaps(n, m) = new(zeros(Int, n, m), zeros(CONST_TYPE, n, m), zeros(Int, n, m), zeros(Int, n, m))
end

function build_op_strings(binops, unaops)
    binops_str = if length(binops) == 0
        ""
    else
        i = 1
        op = binops[i]
        tmp = "\nif operator[i, k] == $i"
        for (i, op) in enumerate(binops)
            if i == 1
                tmp *= "\n    @inbounds cumulator[k, j] = CUDA.cufunc($(op))(cumulator[k, j], array2[k, j])"
            else
                tmp *= "\nelseif operator[i, k] == $i"
                tmp *= "\n    @inbounds cumulator[k, j] = CUDA.cufunc($(op))(cumulator[k, j], array2[k, j])"
            end
        end
        tmp * "\nend"
    end

    unaops_str = if length(unaops) == 0
        ""
    else
        i = 1
        op = unaops[i]
        tmp = "\nif operator[i, k] == $i"
        for (i, op) in enumerate(unaops)
            if i == 1
                tmp *= "\n    @inbounds cumulator[k, j] = CUDA.cufunc($(op))(cumulator[k, j])"
            else
                tmp *= "\nelseif operator[i, k] == $i"
                tmp *= "\n    @inbounds cumulator[k, j] = CUDA.cufunc($(op))(cumulator[k, j])"
            end
        end
        tmp * "\nend"
    end

    return binops_str, unaops_str
end

function build_heap_evaluator(options::Options)
    binops_str, unaops_str = build_op_strings(options.binops, options.unaops)
    return quote
        using CUDA
        function gpuEvalNodes!(i::Int, nrows::Int, nheaps::Int,
                               operator::AbstractArray{Int},
                               constant::AbstractArray{T},
                               feature::AbstractArray{Int}, #0=>use constant
                               degree::AbstractArray{Int}, #0 for degree => stop the tree!
                               cumulator::AbstractArray{T},
                               array2::AbstractArray{T},
                               X::AbstractArray{T}) where {T<:Union{Float32,Float64}}
            index = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
            stride = (blockDim()).x * (gridDim()).x
            n = (nrows*nheaps)

            for _j = index:stride:n
                j = ceil(Int, _j/nheaps)
                k = ((_j - 1) % nheaps) + 1
                if degree[i, k] == 0
                    if feature[i, k] == 0
                        @inbounds cumulator[k, j] = constant[i, k]
                    else
                        @inbounds cumulator[k, j] = X[feature[i,k], j]
                    end
                elseif degree[i, k] == 1
                    $(Meta.parse(unaops_str))
                else
                    $(Meta.parse(binops_str))
                     #Make into if statement.
                end
            end
        end
        function evaluateHeapArrays(i::Int, 
                               operator::AbstractArray{Int},
                               constant::AbstractArray{T},
                               feature::AbstractArray{Int}, #0=>use constant
                               degree::AbstractArray{Int}, #0 for degree => stop the tree!
                               X::AbstractArray{T, 2})::AbstractArray{T, 2} where {T<:Real}
            nfeature = size(X, 1)
            nrows = size(X, 2)
            nheaps = size(operator, 2)
            depth = size(operator, 1)

            if i > depth
                return CUDA.zeros(Float32, nheaps, nrows)
            end
            cumulator = evaluateHeapArrays(2 * i,     operator, constant, feature, degree, X)
            array2    = evaluateHeapArrays(2 * i + 1, operator, constant, feature, degree, X)

            numblocks = ceil(Int, nrows*nheaps/256)

            CUDA.@sync begin
                @cuda threads=256 blocks=numblocks gpuEvalNodes!(i, nrows, nheaps, operator, constant, feature, degree, cumulator, array2, X)
            end
            # Make array of flags for when nan/inf detected.
            # Set the output of those arrays to 0, so they won't give an error.
            return cumulator
        end

        function evaluateHeaps(i::Int, heaps::EquationHeaps,
                               X::AbstractArray{T, 2})::AbstractArray{T, 2} where {T<:Real}
            # Heaps are [node, tree]
            # X is [feature, row]
            # Output is [tree, row]
            operator = cu(heaps.operator)
            constant = cu(T.(heaps.constant))
            feature  = cu(heaps.feature)
            degree   = cu(heaps.degree)
            _X       = cu(X)
            CUDA.@sync begin
                out = evaluateHeapArrays(i, operator, constant, feature, degree, _X)
            end
            return out
        end
    end
end

function compile_heap_evaluator(options::Options)
    func_str = build_heap_evaluator(options)
    Base.MainInclude.eval(func_str)
end

function populate_heap!(heap::EquationHeap, i::Int, tree::Node)::Nothing
    if tree.degree == 0
        heap.degree[i] = 0
        if tree.constant
            heap.constant[i] = tree.val
            return
        else
            heap.feature[i]  = tree.feature
            return
        end
    elseif tree.degree == 1
        heap.degree[i] = 1
        heap.operator[i] = tree.op
        left = 2 * i
        populate_heap!(heap, left, tree.l)
        return
    else
        heap.degree[i] = 2
        heap.operator[i] = tree.op
        left = 2 * i
        right = 2 * i + 1
        populate_heap!(heap, left, tree.l)
        populate_heap!(heap, right, tree.r)
        return
    end
end

function populate_heaps!(heaps::EquationHeaps, i::Int, j::Int, tree::Node)::Nothing
    if tree.degree == 0
        if tree.constant
            heaps.constant[i, j] = tree.val
            return
        else
            heaps.feature[i, j]  = tree.feature
            return
        end
    elseif tree.degree == 1
        heaps.degree[i, j] = 1
        heaps.operator[i, j] = tree.op
        left = 2 * i
        populate_heaps!(heaps, left, j, tree.l)
        return
    else
        heaps.degree[i, j] = 2
        heaps.operator[i, j] = tree.op
        left = 2 * i
        right = 2 * i + 1
        populate_heaps!(heaps, left, j, tree.l)
        populate_heaps!(heaps, right, j, tree.r)
        return
    end
end

function heapify_tree(tree::Node, max_depth::Int)
    max_nodes = 2^(max_depth-1) + 1
    heap = EquationHeap(max_nodes)
    populate_heap!(heap, 1, tree)
    return heap
end

function heapify_trees(trees::Array{Node, 1}, max_depth::Int)
    max_nodes = 2^(max_depth-1) + 1
    num_trees = size(trees, 1)
    heaps = EquationHeaps(max_nodes, num_trees)
    for j=1:num_trees
        populate_heaps!(heaps, 1, j, trees[j])
    end
    return heaps
end

function heapify_trees(trees::Array{Node, 1})
    max_depth = max([countDepth(tree) for tree in trees]...) + 1
    return heapify_trees(trees, max_depth)
end
