using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using BenchmarkTools
using Profile

# using GBT
using GBT: get_gain, grad_hess, grow_tree2, grow_gbt2, SplitInfo, TreeLeaf, Params, predict, find_split_2

# prepare a dataset
data = CSV.read("./data/performance_tot_v2_perc.csv", allowmissing = :auto)
names(data)

features = data[1:53]
X = convert(Array, features)
Y = data[54]
Y = convert(Array{AbstractFloat}, Y)

# idx
idx = zeros(Int, size(X))
@threads for feat in 1:size(X, 2)
    idx[:, feat] = sortperm(view(X, :, feat)) # returns gain value and idx split
end

# initial info
δ, δ² = grad_hess(zeros(size(Y,1)), Y)
∑δ, ∑δ² = sum(δ), sum(δ²)

# set parameters
nrounds = 2
λ = 0.0001
γ = 1e-3
η = 0.1
max_depth = 3
min_weight = 5.0
params1 = Params(nrounds, λ, γ, η, max_depth, min_weight)
params1 = Params(1, λ, γ, 1.0, 6, min_weight)
gain = get_gain(∑δ, ∑δ², params1.λ)

root = TreeLeaf(1, ∑δ, ∑δ², gain, 0.0)
tree = grow_tree2(root, X, idx, δ, δ², params1)
@btime tree = grow_tree2(root, X, δ, δ², params1)

typeof(params1)

# predict - map a sample to tree-leaf prediction
pred = predict(tree, X)
mean((pred .- Y) .^ 2)
# println(sort(unique(pred)))



function test_grow(n, X, idx, δ, δ²)
    for i in 1:n
        root = TreeLeaf(1, ∑δ, ∑δ², gain, 0.0)
        # tree = grow_tree2(root, view(X, :, :), view(idx, :, :), view(δ, :), view(δ², :), params1)
        # tree = grow_tree2(root, view(X, :, :), view(idx, :, :), view(δ, :), view(δ², :), params1)
        tree = grow_tree2(root, X, idx, δ, δ², params1)
    end
end

@time test_grow(1, X, idx, δ, δ²)
@time test_grow(10, X, idx, δ, δ²)
@time test_grow(100, X, idx, δ, δ²)

tree = Tree([root])
grow_tree2!(tree, X, δ, δ², params1)



# find split tests
x1 = X[:, 1]
idx = 1:size(x1, 1)
function test_split(n)
    for i in 1:n
        find_split(x1, δ, δ², ∑δ, ∑δ², λ)
    end
end

function test_split2(n)
    for i in 1:n
        find_split2(x1, idx, δ, δ², ∑δ, ∑δ², λ)
    end
end

function test_split3(n)
    for i in 1:n
        find_split_3(x1, idx, δ, δ², ∑δ, ∑δ², λ)
    end
end
@time test_split(1)
@time test_split2(10000)
@time test_split3(10000)








function grow_gbt2(X::AbstractArray{AbstractFloat}, Y, params::Params)

    μ = mean(Y)
    pred = zeros(size(Y,1)) .* 0
    δ, δ² = grad_hess(pred, Y)
    ∑δ, ∑δ² = sum(δ), sum(δ²)
    gain = get_gain(∑δ, ∑δ², params.λ)

    bias = TreeLeaf(1, 0.0, 0.0, gain, 0.0)
    model = GBTree([bias], params)

    for i in 1:params.nrounds
        # select random rows and cols
        #X, Y = X[row_ids, col_ids], Y[row_ids]
        # get gradients
        δ, δ² = grad_hess(pred, Y)
        ∑δ, ∑δ² = sum(δ), sum(δ²)
        gain = get_gain(∑δ, ∑δ², params.λ)
        # assign a root and grow tree
        root = TreeLeaf(1, ∑δ, ∑δ², gain, 0.0)
        # grow tree
        tree = grow_tree2(root, view(X, :, :), view(δ, :), view(δ², :), params)
        # get update predictions
        pred += predict(tree, X) .* params.η
        # update push tree to model
        push!(model.trees, tree)

        println("iter: ", i, " completed")
    end
    return model
end

function grow_tree2(node::TreeNode, X::AbstractArray, idx::AbstractArray{Int}, δ::AbstractArray, δ²::AbstractArray, params::Params)

    if node.depth < params.max_depth && size(X, 1) >= params.min_weight

        splits = Vector{SplitInfo2}(undef, size(X, 2))
        # idx = zeros(Int, size(X, 1))
        # idx = 1:size(X, 1)

        # idx = zeros(Int, size(X))
        @threads for feat in 1:size(X, 2)
            sortperm!(view(idx, :, feat), view(X, :, feat)) # returns gain value and idx split
        end

        # Search best split for each feature - to be multi-threaded
        @threads for feat in 1:size(X, 2)
        # for feat in 1:size(X, 2)
            # splits[feat] = find_split_2(X[:, feat], δ, δ², node.∑δ, node.∑δ², params.λ) # returns gain value and idx split
            splits[feat] = find_split_2(view(X, :, feat), view(idx, :, feat), view(δ, :), view(δ², :), node.∑δ, node.∑δ², params.λ) # returns gain value and idx split
            splits[feat].feat = feat
        end

        # assign best split
        best = get_max_gain(splits)

        # grow node if best split improve gain
        if best.gain > node.gain + params.γ
            last = size(X, 1)
            node = TreeSplit(
            # grow_tree2(TreeLeaf(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0.0), X, δ, δ², params),
            # grow_tree2(TreeLeaf(node.depth + 1, best.∑δR, best.∑δ²R, best.gainR, 0.0), X, δ, δ², params),
            grow_tree2(TreeLeaf(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0.0), view(X, 1:best.𝑖, :),  view(idx, 1:best.𝑖, :), view(δ, 1:best.𝑖), view(δ², 1:best.𝑖), params),
            grow_tree2(TreeLeaf(node.depth + 1, best.∑δR, best.∑δ²R, best.gainR, 0.0), view(X, best.𝑖+1:last, :), view(idx, best.𝑖+1:last, :), view(δ, best.𝑖+1:last), view(δ², best.𝑖+1:last), params),
            # grow_tree2(TreeLeaf(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0.0), X[1:best.𝑖, :], δ[1:best.𝑖], δ²[1:best.𝑖], params),
            # grow_tree2(TreeLeaf(node.depth + 1, best.∑δR, best.∑δ²R, best.gainR, 0.0), X[best.𝑖+1:last, :], δ[best.𝑖+1:last], δ²[best.𝑖+1:last], params),
            best.feat,
            best.cond)
        end
    end
    if isa(node, TreeLeaf) node.pred = - node.∑δ / (node.∑δ² + params.λ) end
    # node.pred = - node.∑δ / (node.∑δ² + params.λ)
    return node
end


# extract the gain value from the vector of best splits and return the split info associated with best split
function get_max_gain(splits)
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    best.feat = feat
    return best
end

# if we have sum of δ and δ² for whole node. Can fin gain by getting δL += sumδL[𝑖]
# x δ δ² are vectors sorted in ascending order of x value
function find_split_2(x::AbstractArray, idx::AbstractArray{Int}, δ::AbstractArray, δ²::AbstractArray, ∑δ, ∑δ², λ)

    # sortperm!(idx, x)
    # sortperm!(idx, x)
    # idx = 1:size(x, 1)
    # idx = sortperm(x)
    # x = x[idx]
    # δ = δ[idx]
    # δ² = δ²[idx]
    x = view(x, idx)
    δ = view(δ, idx)
    δ² = view(δ², idx)

    gain = get_gain(∑δ, ∑δ², λ)

    best = SplitInfo2(gain, 0.0, 0.0, ∑δ, ∑δ², -Inf, -Inf, 0, 0, 0.0)
    ∑δL, ∑δ²L, ∑δR, ∑δ²R = 0.0, 0.0 , ∑δ, ∑δ²

    𝑖 = 1
    for i in 1:(size(x, 1) - 1)

        ∑δL += δ[i]
        ∑δ²L += δ²[i]
        ∑δR -= δ[i]
        ∑δ²R -= δ²[i]

        if x[i] < x[i+1] # check gain only if there's a change in value
            gainL = get_gain(∑δL, ∑δ²L, λ)
            gainR = get_gain(∑δR, ∑δ²R, λ)
            if gainL + gainR > best.gain
                best.gain = gainL + gainR
                best.∑δL, best.∑δ²L = ∑δL, ∑δ²L
                best.∑δR, best.∑δ²R = ∑δR, ∑δ²R
                best.gainL, best.gainR = gainL, gainR
                best.cond = x[i]
                𝑖 = i
            end
        end
    end
    return best
end
