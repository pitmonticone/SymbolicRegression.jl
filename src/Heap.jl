using SparseArrays
using FromFile
@from "Core.jl" import Node, CONST_TYPE


function evaluateHeaps(i::Int, heaps::EquationHeaps, X::AbstractArray{T, 2}, options::Options)::AbstractArray{T, 2}
	nfeature = size(X, 1)
	nrows = size(X, 2)
	nheaps = size(heaps, 2)

	if i > size(heaps, 1)
		return spzeros(Int, nheaps, nrows)
	end
	left  = evaluateHeaps(2 * i,     heaps, X, options)
	right = evaluateHeaps(2 * i + 1, heaps, X, options)
	f(operator, constant, feature, degree) = (
			convert(T, degree == 0) * (
				convert(T, feature == 0) * constant
				convert(T, feature > 0)  * X[max(feature, 1), :]
			)
			+ convert(T, degree == 1) * options.unaops[operator](_left)
			+ convert(T, degree == 2) * options.binops[operator](_left, _right)
	)
end

mutable struct EquationHeap
    operator::Array{Int, 1}
    constant::Array{CONST_TYPE, 1}
    feature::Array{Int, 1} #0=>use constant
    degree::Array{Int, 1} #0 for degree => stop the tree!

    EquationHeap(n) = new(spzeros(Int, n), spzeros(CONST_TYPE, n), spzeros(Int, n), spzeros(Int, n))
end

mutable struct EquationHeaps
    operator::Array{Int, 2}
    constant::Array{CONST_TYPE, 2}
    feature::Array{Int, 2} #0=>use constant
    degree::Array{Int, 2} #0 for degree => stop the tree!

    EquationHeaps(n, m) = new(spzeros(Int, n, m), spzeros(CONST_TYPE, n, m), spzeros(Int, n, m), spzeros(Int, n, m))
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
