function grow_gbt2(X::AbstractArray, Y, params::Params)

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

function grow_tree2(node::TreeNode, X::AbstractArray, idx::AbstractArray, δ::AbstractArray, δ²::AbstractArray, params::Params)

    if node.depth < params.max_depth && size(X, 1) >= params.min_weight

        splits = Vector{SplitInfo2}(undef, size(X, 2))
        @threads for feat in 1:size(X, 2)
            splits[feat] = SplitInfo2(-Inf, 0.0, 0.0, 0.0, 0.0, -Inf, -Inf, 0, 0, 0.0)
        end

        # idx = zeros(Int, size(X))
        @threads for feat in 1:size(X, 2)
            # sortperm!(view(idx, :, feat), view(X, :, feat)) # returns gain value and idx split
            idx[:, feat] = sortperm(X[:, feat]) # returns gain value and idx split
        end

        # Search best split for each feature - to be multi-threaded
        @threads for feat in 1:size(X, 2)
        # for feat in 1:size(X, 2)
            # splits[feat] = find_split_2(X[:, feat], δ, δ², node.∑δ, node.∑δ², params.λ) # returns gain value and idx split
            find_split!(view(X, view(idx, :, feat), feat), view(δ, view(idx, :, feat)), view(δ², view(idx, :, feat)), node.∑δ, node.∑δ², params.λ, splits[feat]) # returns gain value and idx split
            # find_split!(view(X, view(node.𝑖, view(perm_id, :, feat)), feat), view(δ, view(node.𝑖, view(perm_id, :, feat))) , view(δ², view(node.𝑖, view(perm_id, :, feat))), node.∑δ, node.∑δ², params.λ, splits[feat])
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
            # grow_tree2(TreeLeaf(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0.0), view(X, 1:best.𝑖, :),  view(idx, 1:best.𝑖, :), view(δ, 1:best.𝑖), view(δ², 1:best.𝑖), params),
            # grow_tree2(TreeLeaf(node.depth + 1, best.∑δR, best.∑δ²R, best.gainR, 0.0), view(X, best.𝑖+1:last, :), view(idx, best.𝑖+1:last, :), view(δ, best.𝑖+1:last), view(δ², best.𝑖+1:last), params),

            grow_tree2(TreeLeaf(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0.0), X[view(idx, :, best.feat)[1:best.𝑖], :],  idx[view(idx, :, best.feat)[1:best.𝑖], :], δ[view(idx, :, best.feat)[1:best.𝑖]], δ²[view(idx, :, best.feat)[1:best.𝑖]], params),
            grow_tree2(TreeLeaf(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0.0), X[view(idx, :, best.feat)[best.𝑖+1:last], :],  idx[view(idx, :, best.feat)[best.𝑖+1:last], :], δ[view(idx, :, best.feat)[best.𝑖+1:last]], δ²[view(idx, :, best.feat)[best.𝑖+1:last]], params),

            # grow_tree2(TreeLeaf(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0.0), view(X, view(idx, :, best.feat)[1:best.𝑖], :),  view(idx, view(idx, :, best.feat)[1:best.𝑖], :), view(δ, view(idx, :, best.feat)[1:best.𝑖]), view(δ², view(idx, :, best.feat)[1:best.𝑖]), params),
            # grow_tree2(TreeLeaf(node.depth + 1, best.∑δL, best.∑δ²L, best.gainL, 0.0), view(X, view(idx, :, best.feat)[best.𝑖+1:last], :),  view(idx, view(idx, :, best.feat)[best.𝑖+1:last], :), view(δ, view(idx, :, best.feat)[best.𝑖+1:last]), view(δ², view(idx, :, best.feat)[best.𝑖+1:last]), params),


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
function find_split!(x::AbstractArray, δ::AbstractArray, δ²::AbstractArray, ∑δ, ∑δ², λ, info::SplitInfo2)

    info.gain = get_gain(∑δ, ∑δ², λ)

    # best = SplitInfo2(gain, 0.0, 0.0, ∑δ, ∑δ², -Inf, -Inf, 0, 0, 0.0)
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
            if gainL + gainR > info.gain
                info.gain = gainL + gainR
                info.∑δL, info.∑δ²L = ∑δL, ∑δ²L
                info.∑δR, info.∑δ²R = ∑δR, ∑δ²R
                info.gainL, info.gainR = gainL, gainR
                info.cond = x[i]
                info.𝑖 = i
            end
        end
    end
end
