
# compute the gradient and hessian given target and predict
# linear
function update_grads!(::Val{:linear}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 2}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ[:,1] = 2 * (pred - target) * 𝑤
    @. δ[:,2] = 2 * 𝑤
end

# compute the gradient and hessian given target and predict
# logistic - on linear predictor
function update_grads!(::Val{:logistic}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 2}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ[:,1] = (sigmoid(pred) * (1 - target) - (1 - sigmoid(pred)) * target) * 𝑤
    @. δ[:,2] = sigmoid(pred) * (1 - sigmoid(pred)) * 𝑤
end

# compute the gradient and hessian given target and predict
# poisson
# Reference: https://isaacchanghau.github.io/post/loss_functions/
function update_grads!(::Val{:poisson}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 2}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ[:,1] = (exp(pred) - target) * 𝑤
    @. δ[:,2] = exp(pred) * 𝑤
end

function logit(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = log(x / (1 - x))
    return x
end

function logit(x::T) where T <: AbstractFloat
    x = log(x / (1 - x))
    return x
end

function sigmoid(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = 1 / (1 + exp(-x))
    return x
end

function sigmoid(x::T) where T <: AbstractFloat
    x = 1 / (1 + exp(-x))
    return x
end

# update the performance tracker
function update_track!(track::SplitTrack{T}, λ::T) where T <: AbstractFloat
    track.gainL = (track.∑δL[1] ^ 2 / (track.∑δL[2] + λ * track.∑𝑤L)) / 2
    track.gainR = (track.∑δR[1] ^ 2 / (track.∑δR[2] + λ * track.∑𝑤R)) / 2
    track.gain = track.gainL + track.gainR
end

# Calculate the gain for a given split
function get_gain(∑δ::Vector{T}, ∑𝑤::T, λ::T) where T <: AbstractFloat
    gain = (∑δ[1] ^ 2 / (∑δ[2] + λ * ∑𝑤)) / 2
    return gain
end
