using CUDA
using FromFile
@from "src/Core.jl" import Node, CONST_TYPE, Options
@from "src/MutationFunctions.jl" import genRandomTree
@from "src/Heap.jl" import EquationHeap, EquationHeaps, build_op_strings, build_heap_evaluator, compile_heap_evaluator, populate_heap!, populate_heaps!, heapify_tree, heapify_trees

X = CUDA.randn(Float32, 5, 300)
options = Options(binary_operators=(+, *, -, /),
                  unary_operators=(cos, exp),
                  npopulations=30);
# function genRandomTree(length::Int, options::Options, nfeatures::Int)::Node


compile_heap_evaluator(options)

using BenchmarkTools

trees = [genRandomTree(8, options, 5) for i=1:300];

heaps = heapify_trees(trees[1:100], 10);
@btime evaluateHeaps(1, heaps, X);
heaps = heapify_trees(trees[1:end], 10);
@btime evaluateHeaps(1, heaps, X);


# # Here is how you declare a function GPU-ready:
# myop(x) = cos(x)
# CUDA.@cufunc myop(x) = cos(x)
# # Heap stuff
# using CUDA
# using FromFile
# @from "src/Core.jl" import Node, CONST_TYPE, Options
# @from "src/Heap.jl" import EquationHeap, EquationHeaps, build_op_strings, build_heap_evaluator, compile_heap_evaluator, populate_heap!, populate_heaps!, heapify_tree, heapify_trees, evaluateHeaps

# X = randn(Float32, 5, 300)
# options = Options(binary_operators=(+, *, -, /),
                  # unary_operators=(cos, exp),
                  # npopulations=30);
# tree = cos(Node(1) * 3f0 - Node(3)) + 2f0

# heaps = heapify_trees([tree], 10)

# evaler = build_heap_evaluator(options)
# # compile_heap_evaluator(options)

# # Here is how you declare a function GPU-ready:
# myop(x) = cos(x)
# CUDA.@cufunc myop(x) = cos(x)

# function gpuEvalNodes!(i::Int, nrows::Int, nheaps::Int, cumulator::AbstractArray{Float32})
    # index = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    # stride = (blockDim()).x * (gridDim()).x
    # n = (nrows*nheaps)


    # for _j = index:stride:n
        # j = ceil(Int, _j/nheaps)
        # k = ((_j - 1) % nheaps) + 1
        # # for k = index_k:stride_k:nheaps
        # @inbounds cumulator[k, j] = CUDA.cufunc(cos)(cumulator[k, j])
        # # end
    # end
# end

# function gpuEvalNodes!(i::Int, nrows::Int, nheaps::Int, operator::AbstractArray{Int, 2},
                       # constant::AbstractArray{CONST_TYPE, 2}, feature::AbstractArray{Int, 2},
                       # degree::AbstractArray{Int, 2}, cumulator::AbstractArray{Float32},
                       # array2::AbstractArray{Float32})
    # index = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    # stride = (blockDim()).x * (gridDim()).x
    # n = (nrows*nheaps)
    # for _j = index:stride:n
        # j = ceil(Int, _j/nheaps)
        # k = ((_j - 1) % nheaps) + 1

        # o = operator[i, k]
        # c = constant[i, k]
        # f = feature[i, k]
        # x = X[max(f, 1), j]
        # d = degree[i, k]
        # l = cumulator[k, j]
        # r = array2[k, j]
        # out = if d == 0
                # if f == 0
                    # T(c)
                # else
                    # x
                # end
            # elseif  d == 1
                # if o == 1
                    # cos(l)
                # elseif  o == 2
                    # exp(l)
                # end
            # else
                # if o == 1
                    # l + r
                # elseif  o == 2
                    # l * r
                # elseif  o == 3
                    # l - r
                # elseif  o == 4
                    # l / r
                # end
            # end
        # @inbounds cumulator[k, j] = out
    # end
# end

# operator = cu(heaps.operator); constant = cu(heaps.constant); feature = cu(heaps.feature); degree = cu(heaps.degree)
# nfeature = size(X, 1); nrows = size(X, 2); nheaps = size(operator, 2); depth = size(operator, 1)
# i = 1
# # cumulator = evaluateHeaps(2 * i, heaps, X)
# # array2    = evaluateHeaps(2 * i + 1, heaps, X)
# cumulator   = CUDA.randn(Float32, nheaps, nrows)
# array2      = CUDA.randn(Float32, nheaps, nrows)
# numblocks = ceil(Int, nheaps*nrows/256)

# CUDA.@sync begin
    # @cuda threads=256 blocks=numblocks gpuEvalNodes!(i, nrows, nheaps, cumulator)
# end








# # SR stuff
# using SymbolicRegression
# X = randn(Float32, 5, 300)
# f = (x,) -> 2 * abs(cos(x[3])) ^ 3.2 - 0.3 * abs(x[1]) ^ 2.1 + 0.4 * abs(x[1]) ^ 1.5
# y = [f(X[:, i]) for i=1:300]
# options = SymbolicRegression.Options(binary_operators=(+, *, -, /, ^),
                                     # unary_operators=(cos,),
                                     # constraints=(cos=>5, (^)=>(-1, 3)),
                                     # npopulations=128 * 4,
                                     # ncyclesperiteration=3000,
                                     # useFrequency=true,
                                     # maxsize=50)

# hallOfFame = EquationSearch(X, y;
                            # niterations=1000, options=options,
                            # numprocs=128)








# # CUDA stuff:

# using SymbolicRegression
# using SymbolicRegression: Dataset, EvalLoss
# using CUDA
# using BenchmarkTools

# N = 1000
# Xhost = randn(Float32, 5, N)
# yhost = 2 .* cos.(Xhost[3, :])
# X = CUDA.randn(Float32, 5, N)
# y = 2 .* cos.(X[3, :])
# dataset = Dataset(X, y)

# options = SymbolicRegression.Options(binary_operators=(+, *, -, /),
                                     # unary_operators=(cos, exp),
                                     # npopulations=30);
# # hallOfFame = EquationSearch(X, y; niterations=5, options=options)
# tree = cos(Node(1) * 3f0 - Node(3)) + 2f0
# tree *= tree
# tree *= tree
# printTree(tree, options)
# @btime begin
    # for i=1:100
        # t = evalTreeArray(tree, X, options);
    # end
# end
# @btime begin
    # for i=1:100
        # t = evalTreeArray(tree, Xhost, options)
    # end
# end


# # using CUDA, BenchmarkTools, Statistics
# # for N in [1000, 10000]
    # # c1 = CUDA.ones(Float32, N)
    # # c2 = ones(Float32, N)
    # # res1 = @benchmark cos.($c1);
    # # # res1 = @benchmark cos.($c1);
    # # res2 = @benchmark cos.($c2);
    # # println("Size $N: CUDA=$(median(res1)); CPU=$(median(res2))")
# # end


# # for N in [1000, 10000] Xhost = randn(Float32, N) X = CUDA.randn(Float32, N)
    # # res1 = @btime cos.($X);
    # # res2 = @btime cos.($Xhost);
    # # println("Size $N: CUDA=$(median(res1)); CPU=$(median(res2))")
# # end
    # # @btime CUDA.@sync blocking=false cos.($X); @btime cos.($Xhost);

# function add_broadcast!(y, x)
    # CUDA.@sync y .+= x
    # return
# end














# using CUDA



# function f(x::AbstractArray{T}, y::AbstractArray{T})::AbstractArray{T} where {T<:Real}
    # y .= x .+ y
# end

# x = CUDA.ones(10)
# y = CUDA.ones(10)
# xhost = ones(10)
# yhost = ones(10)

# xsparse = spzeros(10)



