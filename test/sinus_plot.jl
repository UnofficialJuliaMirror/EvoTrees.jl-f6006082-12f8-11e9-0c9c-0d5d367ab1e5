using BenchmarkTools
using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Plots

using Revise
using EvoTrees
using EvoTrees: sigmoid, logit

# prepare a dataset
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# set parameters
loss = :linear
nrounds = 200
λ = 0.5
γ = 0.5
η = 0.1
max_depth = 5
min_weight = 1.0
rowsample = 0.5
colsample = 1.0
nbins = 100

# linear
params1 = EvoTreeRegressor(
    loss=:linear,
    nrounds=200, nbins = 100,
    λ = 0.5, γ=0.5, η=0.1,
    max_depth = 5, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric=:mae)
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 10, metric=:mae)
@time pred_train_linear = predict(model, X_train)
@time pred_eval_linear = predict(model, X_eval)
mean(abs.(pred_train_linear .- Y_train))
sqrt(mean((pred_train_linear .- Y_train) .^ 2))

# logistic / cross-entropy
params1 = EvoTreeRegressor(
    loss=:logistic,
    nrounds=200, nbins = 100,
    λ = 0.5, γ=0.5, η=0.1,
    max_depth = 5, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :logloss)
@time pred_train_logistic = predict(model, X_train)
@time pred_eval_logistic = predict(model, X_eval)
sqrt(mean((pred_train_logistic .- Y_train) .^ 2))

# Poisson
params1 = EvoTreeRegressor(
    loss=:poisson,
    nrounds=200, nbins = 100,
    λ = 0.5, γ=0.5, η=0.1,
    max_depth = 5, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :logloss)
@time pred_train_poisson = predict(model, X_train)
@time pred_eval_poisson = predict(model, X_eval)
sqrt(mean((pred_train_poisson .- Y_train) .^ 2))

x_perm = sortperm(X_train[:,1])
plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "gray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train_linear[x_perm], color = "navy", linewidth = 1.5, label = "Linear")
plot!(X_train[:,1][x_perm], pred_train_logistic[x_perm], color = "darkred", linewidth = 1.5, label = "Logistic")
plot!(X_train[:,1][x_perm], pred_train_poisson[x_perm], color = "green", linewidth = 1.5, label = "Poisson")
savefig("regression_sinus.png")

###############################
## Quantiles
###############################
# q50
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.5,
    nrounds=200, nbins = 100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric=:quantile)
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 10, metric=:mae)
@time pred_train_q50 = predict(model, X_train)

# q20
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.2,
    nrounds=200, nbins = 100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :quantile)
@time pred_train_q20 = predict(model, X_train)

# q80
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.8,
    nrounds=200, nbins = 100,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :quantile)
@time pred_train_q80 = predict(model, X_train)

x_perm = sortperm(X_train[:,1])
plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "gray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train_q50[x_perm], color = "navy", linewidth = 1.5, label = "Median")
plot!(X_train[:,1][x_perm], pred_train_q20[x_perm], color = "darkred", linewidth = 1.5, label = "Q20")
plot!(X_train[:,1][x_perm], pred_train_q80[x_perm], color = "green", linewidth = 1.5, label = "Q80")
savefig("quantiles_sinus.png")


###############################
## Quantiles v2
###############################

# prepare a dataset
features = rand(1_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = rand(1_000) .* 1
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# q50
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.5,
    nrounds=1, nbins = 100,
    λ = 0.5, γ=0.1, η=1.0,
    max_depth = 1, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric=:quantile)
@time pred_train_q50 = predict(model, X_train)
sum(pred_train_q50 .< Y_train) / length(Y_train)

# q20
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.2,
    nrounds=1, nbins = 100,
    λ = 0.5, γ=0.1, η=1.0,
    max_depth = 1, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :quantile)
@time pred_train_q20 = predict(model, X_train)
sum(pred_train_q20 .< Y_train) / length(Y_train)

# q80
params1 = EvoTreeRegressor(
    loss=:quantile, α=0.75,
    nrounds=1, nbins = 100,
    λ = 0.5, γ=0.1, η=1.0,
    max_depth = 1, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 10, metric = :quantile)
@time pred_train_q80 = predict(model, X_train)

x_perm = sortperm(X_train[:,1])
plot(X_train, Y_train, ms = 1, mcolor = "gray", mscolor = "gray", background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train_q50[x_perm], color = "navy", linewidth = 1.5, label = "Median")
plot!(X_train[:,1][x_perm], pred_train_q20[x_perm], color = "darkred", linewidth = 1.5, label = "Q20")
plot!(X_train[:,1][x_perm], pred_train_q80[x_perm], color = "green", linewidth = 1.5, label = "Q80")
savefig("quantiles_sinus.png")
