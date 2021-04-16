module RegressionToolsR

using GLM
using DataFrames
import Statistics: quantile, var, mean
import LinearAlgebra: inv, diag, PosDefException
import Distributions: TDist, UnivariateDistribution
using StatsBase

export predint, rstudent, stepwise, step, glm_step, glm_stepwise

"""
Predict the linear response of `xf` when using given *strictly* linear model. Model
    must have been created using a DataFrame

    predint(mdl, xf[, p])

# Arguments:
- `mdl::RegressionModel`: A strictly linear model with no transform on the formula.
- `xf::AbstractArray`: An array of locations on the x-axis to forecast
- `p::Float64=0.95`: The certainty we want to have (e.g. default = 0.95, which gives
        a 95% confidence interval)

# Returns:
- `pred::Array{Union{Missing, Float64}}((n,))`: Predicted values for given xf,
    `n` is the length of `xf`.
- `dev::Array{Float64}((n,))`: Deviation from the prediction that the interval has.

# Examples
```jldoctest
julia> using DataFrames, GLM, RegressionToolsR
julia> n = 10; x = rand(n); y = 4*x+0.05*rand(n);
julia> df = DataFrame(X = x, Y = y); fm = @formula(Y ~ X);
julia> mdl = lm(fm, df);
julia> yf, dev = predint(mdl, 0:0.5:10);
julia> using Plots
julia> plot((xf, yf), ribbon=dev, fillalpha=0.25, lab="Prediction", linewidth=3);
julia> scatter!((x, y), lab="Data");
julia> title!("Example usage of predint");
```
See also: [`confint`](@ref)
"""
function predint(mdl::RegressionModel, xf::AbstractArray; p::Float64=0.95)
    # TODO: Ensure that `mod` is purely linear and of one variable
    # Convert `p` into a quantile-compatible format
    p = 0.5 + p/2;
    x = mdl.model.pp.X[:,2]; # Get the vector
    y = mdl.model.rr.y;
    n = length(x); # Get the dimension of the dataset
    x_sym = mdl.mf.f.rhs.terms[2].sym; # Get the symbols
    y_sym = mdl.mf.f.lhs.sym;;
    s2 = var(y); # Calculate the sample variance of y
    sx2 = var(x); # Calculate the sample variance of x
    xbar = mean(x); # Calculate the sample mean in x
    # Calculate the standard error with prediction intervals
    sse = sqrt.(s2 .* ((1 + (1/n)) .+ (xf .- xbar).^2 ./ ((n-1)*sx2)));
    # Get the t-value for the given p
    t_p = quantile(TDist(n), p);
    # Get the forecasted values
    pred = predict(mdl, eval(:(DataFrame($x_sym = $xf))));
    # Calculate the deviance for the prediction interval
    dev = t_p*sse;
    # Return all the appropriate values
    return pred, dev
end

"""
Calculate the studentized residuals of a model `mdl`

    rstudent(mdl)

# Arguments
  - `mdl::RegressionModel`: A regression model compatible with the GLM package

# Returns
  - `s_res::Array{Float64}((n,))`: The studentized residual for each predicted value

# Examples
```jldoctest
julia> using DataFrames, GLM, RegressionToolsR
julia> n = 10; x = rand(n); y = 4*x+0.05*rand(n);
julia> df = DataFrame(X = x, Y = y); fm = @formula(Y ~ X);
julia> mdl = lm(fm, df);
julia> s_res = rstudent(mdl);
julia> using Plots;
julia> qqnorm(s_res);
```

See also: [`residuals`](@ref)
"""
function rstudent(mdl::RegressionModel)
    X = mdl.model.pp.X; # Get the data matrix
    # Calculate the Least Squares projection to get leverage
    h = diag(X*inv(transpose(X)*X)*transpose(X));
    x = X[:,2]; # Get data vector
    n = length(x); # Get data size
    # Calculate residuals
    res = mdl.model.rr.y - mdl.model.rr.mu;
    # Calculate delta_i values
    di = res.^2 ./ (1 .- h);
    # Calculate Out-of-Sample variance
    s_hat_sq = (sum(res.^2) .- di) ./ (n-3);
    # Calculate studentized residuals
    s_res = res ./ sqrt.(s_hat_sq .* (1 .- h));
    return s_res
end

# Credit to https://stackoverflow.com/questions/49794476/backward-elimination-forward-selection-in-multilinear-regression-in-julia

"""
Create a formula for linear regression from a number of Symbols
    compose(lhs, rhs)

# Arguments
    - `lhs::Symbol`: The dependent/response variable
    - `rhs::AbstractVector{Symbol}`: The independent/regressor variables

# Returns
    `fm::FormulaTerm`: A formula representing the regression
"""
function compose(lhs::Symbol, rhs::AbstractVector{Symbol})::FormulaTerm
    return Term(lhs) ~ sum([ConstantTerm(1), Term.(rhs)...]);
end

function compose(lhs::Symbol, rhs::AbstractVector{AbstractTerm})::FormulaTerm
    return Term(lhs) ~ sum([ConstantTerm(1), rhs...]);
end

function compose(lhs::Symbol, rhs::AbstractTerm)::FormulaTerm
    return Term(lhs) ~ ConstantTerm(1) + rhs;
end

"""
Perform one step of stepwise regression
    step(df, lhs, rhs, forward, use_aic)

# Arguments
    - `df::DataFrame`: DataFrame we are using
    - `lhs::Symbol`: The dependent/response variable
    - `rhs::AbstractVector{Symbol}`: The current used regressor variables
    - `forward::Bool`: Whether we use forward or backward steps
    - `use_aic::Bool`: Whether to use `aic` or `bic`

# Returns
    - `best_rhs::Array{Symbol,1}`: A new `rhs` with at most one more regressor
    - `improved::Bool`: Whether any new regression variable was added
"""
function step(df::DataFrame, lhs::Symbol, rhs::AbstractVector{Symbol},
              forward::Bool, use_aic::Bool)::Tuple{Array{Symbol,1},Bool}
    options = forward ? setdiff(Symbol.(names(df)), [lhs; rhs]) : rhs
    fun = use_aic ? aic : bic
    if(isempty(options))
        return (rhs, false)
    end
    best_fun = fun(lm(compose(lhs, rhs), df))
    improved = false
    best_rhs = rhs
    for opt in options
        this_rhs = forward ? [rhs; opt] : setdiff(rhs, [opt])
        fm = compose(lhs, this_rhs);
        mdl = lm(fm, df);
        this_fun = fun(mdl);
        if this_fun < best_fun
            best_fun = this_fun
            best_rhs = this_rhs
            improved = true
        end
    end
    return best_rhs, improved
end


"""
Perform one step of stepwise regression
    step(df, lhs, rhs, forward, use_aic)

# Arguments
    - `df::DataFrame`: DataFrame we are using
    - `lhs::Symbol`: The dependent/response variable
    - `rhs::Symbol`: The current used regressor variable
    - `forward::Bool`: Whether we use forward or backward steps
    - `use_aic::Bool`: Whether to use `aic` or `bic`

# Returns
    - `best_rhs::Array{Symbol,1}`: A new `rhs` with at most one more regressor
    - `improved::Bool`: Whether any new regression variable was added
"""
function step(df::DataFrame, lhs::Symbol, rhs::Symbol,
              forward::Bool, use_aic::Bool)::Tuple{Array{Symbol,1},Bool}
    return step(df, lhs, [rhs], forward, use_aic);
end


"""
Perform stepwise regression on a dataframe, similar to stepAIC or stepBIC in R

    `stepwise(df, lhs[, forward = true, use_aic = true])`

# Arguments
    - `df::DataFrame`: The DataFrame that is being modeled
    - `lhs::Symbol`: A symbolic version of the column being regressed on
    - `forward::Bool`: *optional* Whether to do forward or backward regression
    - `use_aic::Bool`: *optional* Whether to use `aic` function or `bic` function.
# Returns
    - `mdl::RegressionModel`: A linear model representing the best terms to use

# Examples
```jldoctest
julia> using DataFrames, GLM, RegressionToolsR, RDatasets
julia> df = dataset("datasets", "swiss")[:,2:end]
julia> mdl = stepwise(df, :Fertility);
```

See also: [`lm`](@ref), [`aic`](@ref), [`bic`](@ref)
"""
function stepwise(df::DataFrame, lhs::Symbol; forward::Bool=true,
                  use_aic::Bool=true)::RegressionModel
    rhs = forward ? Symbol[] : setdiff(propertynames(df), [lhs])
    while true
        rhs, improved = step(df, lhs, rhs, forward, use_aic)
        if(!improved)
            return lm(compose(lhs, rhs), df)
        end
    end
end

"""
Perform one step of stepwise regression on general model
    `glm_step(df, lhs, rhs, family, link, forward, use_aic)`

# Arguments
    - `df::DataFrame`: DataFrame we are using
    - `lhs::Symbol`: The dependent/response variable
    - `rhs::AbstractVector{Symbol}`: The current used regressor variables
    - `family::UnivariateDistribution`: Distribution to use (must be supported by GLM.jl)
    - `link::GLM.Link`: Link function to use
    - `forward::Bool`: Whether we use forward or backward steps
    - `use_aic::Bool`: Whether to use `aic` or `bic`

# Returns
    - `best_rhs::Array{Symbol,1}`: A new `rhs` with at most one more regressor
    - `improved::Bool`: Whether any new regression variable was added
"""
function glm_step(df::DataFrame, lhs::Symbol, rhs::AbstractVector{Symbol},
                  family::UnivariateDistribution, link::GLM.Link,
                  forward::Bool, use_aic::Bool)::Tuple{Vector{Symbol},Bool}
    options = forward ? setdiff(Symbol.(names(df)), [lhs; rhs]) : rhs
    fun = use_aic ? aic : bic
    if(isempty(options))
        return (rhs, false)
    end
    best_fun = fun(glm(compose(lhs, rhs), df, family, link))
    improved = false
    best_rhs = rhs
    for opt in options
        this_rhs = forward ? [rhs; opt] : setdiff(rhs, [opt])
        fm = compose(lhs, this_rhs);
        mdl = glm(fm, df, family, link);
        this_fun = fun(mdl);
        if this_fun < best_fun
            best_fun = this_fun
            best_rhs = this_rhs
            improved = true
        end
    end
    return best_rhs, improved
end

"""
Perform one step of stepwise regression
    glm_step(df, lhs, rhs, family, link, forward, use_aic)

# Arguments
    - `df::DataFrame`: DataFrame we are using
    - `lhs::Symbol`: The dependent/response variable
    - `rhs::Symbol`: The current used regressor 
    - `family::UnivariateDistribution`: Distribution to use (must be supported by GLM.jl)
    - `link::GLM.Link`: Link function to use
    - `forward::Bool`: Whether we use forward or backward steps
    - `use_aic::Bool`: Whether to use `aic` or `bic`

# Returns
    - `best_rhs::Array{Symbol,1}`: A new `rhs` with at most one more regressor
    - `improved::Bool`: Whether any new regression variable was added
"""
function glm_step(df::DataFrame, lhs::Symbol, rhs::Symbol,
                  family::UnivariateDistribution, link::GLM.Link,
                  forward::Bool, use_aic::Bool)::Tuple{Vector{Symbol},Bool}
    return glm_step(df, lhs, [rhs], family, link, forward, use_aic);
end

function interact_fwd_step(df::DataFrame, lhs::Symbol, rhs::AbstractVector{AbstractTerm},
                       family::UnivariateDistribution, link::GLM.Link, use_aic::Bool)::Tuple{Vector{AbstractTerm},Bool}
    reg_terms = term.(setdiff(propertynames(df), [lhs]));
    options = vcat(reg_terms, [reg_terms[i]&reg_terms[j] for i in 1:length(reg_terms)
                                                         for j in (i+1):length(reg_terms)]);
    fun = use_aic ? aic : bic
    if(isempty(options))
        return rhs, false;
    end
    best_fun = fun(glm(compose(lhs, rhs), df, family, link))
    improved = false
    best_rhs = rhs
    for opt in options
        this_rhs = [rhs; opt];
        fm = compose(lhs, this_rhs);
        println(this_rhs);
        mdl = glm(fm, df, family, link);
        this_fun = fun(mdl);
        if this_fun < best_fun
            best_fun = this_fun
            best_rhs = this_rhs
            improved = true
        end
    end
    return best_rhs, improved
end

"""
Perform GLM stepwise regression on a dataframe, similar to stepAIC or stepBIC in R

    `glm_stepwise(df, lhs, family, link; forward = true, use_aic = true)`

# Arguments
    - `df::DataFrame`: The DataFrame that is being modeled
    - `lhs::Symbol`: A symbolic version of the column being regressed on
    - `family::UnivariateDistribution`: Distribution to use (must be supported by GLM.jl)
    - `link::GLM.Link`: Link function to use
    - `forward::Bool`: *optional* Whether to do forward or backward regression
    - `use_aic::Bool`: *optional* Whether to use `aic` function or `bic` function.
# Returns
    - `mdl::RegressionModel`: A generalized linear model representing the best terms to use

# Examples
WORK IN PROGRESS

See also: [`glm`](@ref), [`lm`](@ref), [`aic`](@ref), [`bic`](@ref)
"""
function glm_stepwise(df::DataFrame, lhs::Symbol, family::UnivariateDistribution,
                      link::GLM.Link; forward::Bool=true, use_aic::Bool=true, interact::Bool=false)::RegressionModel
    if !forward && interact
        error("Does not support interaction with backward stepwise!");
    end
    rhs = forward ? Symbol[] : setdiff(propertynames(df), [lhs])
    improved = false;
    while true
        rhs, improved = glm_step(df, lhs, rhs, family, link, forward, use_aic)
        if(!improved)
            return glm(compose(lhs, rhs), df, family, link);
        end
    end
end
end # module
