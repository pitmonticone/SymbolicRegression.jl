# Heap stuff
using CUDA
using FromFile
@from "src/Core.jl" import Node, CONST_TYPE, Options
@from "src/Heap.jl" import EquationHeap, EquationHeaps, build_op_strings, build_heap_evaluator, compile_heap_evaluator, populate_heap!, populate_heaps!, heapify_tree, heapify_trees, evaluateHeaps

X = randn(Float32, 5, 300)
options = Options(binary_operators=(+, *, -, /),
                  unary_operators=(cos, exp),
                  npopulations=30);
tree = cos(Node(1) * 3f0 - Node(3)) + 2f0

heaps = heapify_trees([tree], 10)

compile_heap_evaluator(options)


operator = CuArray(heaps.operator)
constant = CuArray(heaps.constant)
feature  = CuArray(heaps.feature)
degree   = CuArray(heaps.degree)

nfeature = size(X, 1)
nrows = size(X, 2)
nheaps = size(operator, 2)
depth = size(operator, 1)

i = 1
# cumulator = evaluateHeaps(2 * i, heaps, X)
# array2    = evaluateHeaps(2 * i + 1, heaps, X)
cumulator = CUDA.randn(Float32, nheaps, nrows)
array2 = CUDA.randn(Float32, nheaps, nrows)

#nrows, nheaps
numblocks_x = ceil(Int, nrows/256)
numblocks_y = ceil(Int, nheaps/256)

CUDA.@sync begin
    @cuda threads=(256, 256) blocks=(numblocks_x, numblocks_y) gpuEvalNodes!(i, nrows, nheaps,
                                                                             operator, constant, feature,
                                                                             degree, cumulator, array2)
end








# SR stuff
using SymbolicRegression
X = randn(Float32, 5, 300)
f = (x,) -> 2 * abs(cos(x[3])) ^ 3.2 - 0.3 * abs(x[1]) ^ 2.1 + 0.4 * abs(x[1]) ^ 1.5
y = [f(X[:, i]) for i=1:300]
options = SymbolicRegression.Options(binary_operators=(+, *, -, /, ^),
                                     unary_operators=(cos,),
                                     constraints=(cos=>5, (^)=>(-1, 3)),
                                     npopulations=128 * 4,
                                     ncyclesperiteration=3000,
                                     useFrequency=true,
                                     maxsize=50)

hallOfFame = EquationSearch(X, y;
                            niterations=1000, options=options,
                            numprocs=128)








# CUDA stuff:

using SymbolicRegression
using SymbolicRegression: Dataset, EvalLoss
using CUDA
using BenchmarkTools

N = 1000
Xhost = randn(Float32, 5, N)
yhost = 2 .* cos.(Xhost[3, :])
X = CUDA.randn(Float32, 5, N)
y = 2 .* cos.(X[3, :])
dataset = Dataset(X, y)

options = SymbolicRegression.Options(binary_operators=(+, *, -, /),
                                     unary_operators=(cos, exp),
                                     npopulations=30);
# hallOfFame = EquationSearch(X, y; niterations=5, options=options)
tree = cos(Node(1) * 3f0 - Node(3)) + 2f0
tree *= tree
tree *= tree
printTree(tree, options)
@btime begin
    for i=1:100
        t = evalTreeArray(tree, X, options);
    end
end
@btime begin
    for i=1:100
        t = evalTreeArray(tree, Xhost, options)
    end
end


# using CUDA, BenchmarkTools, Statistics
# for N in [1000, 10000]
    # c1 = CUDA.ones(Float32, N)
    # c2 = ones(Float32, N)
    # res1 = @benchmark cos.($c1);
    # # res1 = @benchmark cos.($c1);
    # res2 = @benchmark cos.($c2);
    # println("Size $N: CUDA=$(median(res1)); CPU=$(median(res2))")
# end


# for N in [1000, 10000] Xhost = randn(Float32, N) X = CUDA.randn(Float32, N)
    # res1 = @btime cos.($X);
    # res2 = @btime cos.($Xhost);
    # println("Size $N: CUDA=$(median(res1)); CPU=$(median(res2))")
# end
    # @btime CUDA.@sync blocking=false cos.($X); @btime cos.($Xhost);

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end














using CUDA



function f(x::AbstractArray{T}, y::AbstractArray{T})::AbstractArray{T} where {T<:Real}
    y .= x .+ y
end

x = CUDA.ones(10)
y = CUDA.ones(10)
xhost = ones(10)
yhost = ones(10)

xsparse = spzeros(10)



