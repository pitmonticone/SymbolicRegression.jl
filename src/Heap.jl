using SparseArrays
using FromFile
@from "Core.jl" import Node, CONST_TYPE


function build_op_strings(binops, unaops)
    binops_str = if length(binops) == 0
        "\nT(0)"
    else
        i = 1
        op = binops[i]
        tmp = "\nif o == $i"
        for (i, op) in enumerate(binops)
            if i == 1
                tmp *= "\n    $(op)(l, r)"
            else
                tmp *= "\nelseif o == $i"
                tmp *= "\n    $(op)(l, r)"
            end
        end
        tmp * "\nend"
    end

    unaops_str = if length(unaops) == 0
        "\nT(0)"
    else
        i = 1
        op = unaops[i]
        tmp = "\nif o == $i"
        for (i, op) in enumerate(unaops)
            if i == 1
                tmp *= "\n    $(op)(l)"
            else
                tmp *= "\nelseif o == $i"
                tmp *= "\n    $(op)(l)"
            end
        end
        tmp * "\nend"
    end

    return binops_str, unaops_str
end

function build_heap_evaluator(options::Options)
    binops_str, unaops_str = build_op_heap(options.binops, options.unaops)
    return quote
        function gpuEvalNodes!(i::Int, nrows::Int, nheaps::Int,
                               cumulator::AbstractArray{T}, array2::AbstractArray{T})
            #for j=1:nrows
                #for k=1:nheaps
            
            index_j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            stride_j = blockDim().x * gridDim().x

            index_k = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            stride_k = blockDim().y * gridDim().y

            for j=index_j:stride_j:nrows
                for k=index_k:stride_k:nheaps
                    x = X[max(feature, 1), j]::T
                    o = operators[i, k]::Int
                    c = T(constants[i, k])
                    f = features[i, k]::Int
                    d = degrees[i, k]::Int
                    l = cumulator[k, j]::T
                    r = array2[k, j]::T
                    out = if d == 0
                            if f == 0
                                c
                            else
                                x
                            end
                        elseif d == 1
                            $(Meta.parse(unaops_str))
                        else
                            $(Meta.parse(binops_str))
                             #Make into if statement.
                        end
                    @inbounds cumulator[k, j] = out
                end
            end
        end

        function evaluateHeaps(i::Int, heaps::EquationHeaps,
                               X::AbstractArray{T, 2})::AbstractArray{T, 2}
            # Heaps are [node, tree]
            # X is [feature, row]
            # Output is [tree, row]
            nfeature = size(X, 1)
            nrows = size(X, 2)
            nheaps = size(heaps, 2)

            if i > size(heaps, 1)
                return CUDA.zeros(Int, nheaps, nrows)
            end
            cumulator = evaluateHeaps(2 * i, heaps, X)
            array2 =    evaluateHeaps(2 * i + 1, heaps, X)
            CUDA.@sync begin
                @cuda threads=(256,256) blocks=(numblocks2,numblocks2) gpuEvalNodes!(cumulator, array2)
            end
            # Make array of flags for when nan/inf detected.
            # Set the output of those arrays to 0, so they won't give an error.
            return cumulator
        end
    end
end

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
        heaps.degree[i] = 1
        heaps.operator[i] = tree.op
        left = 2 * i
        populate_heaps!(heaps, left, j, tree.l)
        return
    else
        heaps.degree[i] = 2
        heaps.operator[i] = tree.op
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
