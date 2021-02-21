module RegressionToolsR

import GLM
import DataFrames
import LinearAlgebra
import Distributions

"""
Predict the linear response of `xf` when using formula `fm` on DataFrame `df`,
  then return the prediction interval of probability `p`.

    predint(df, fm, xf[, p])

# Arguments:
- `df::DataFrame`: A DataFrame with the regressor and regressee in it as columns
- `fm::Formula`: A formula you would use with GLM for a LinearModel
- `xf::AbstractArray`: An array of locations on the x-axis to forecast
- `p::Float64=0.95`: The certainty we want to have (e.g. default = 0.95, which gives
        a 95% confidence interval)

# Returns:
- `pred::Array{Union{Missing, Float64}}((n,))`: Predicted values for given xf,
    `n` is the length of `xf`.
- `interval::Array{Float64}((n, 2))`: For each point, give the prediction interval.
- `dev::Array{Float64}((n,))`: Deviation from the prediction that the interval has.

# Examples
```jldoctest
julia> using DataFrames, GLM
julia> n = 10; x = rand(n); y = 4*x+0.05*rand(n);
julia> df = DataFrame(X = x, Y = y); fm = @formula(Y ~ X);
julia> yf, interval, dev = predint(df, fm, 0:0.5:10);
julia> using Plots
julia> plot((xf, yf), ribbon=dev, fillalpha=0.25, lab="Prediction", linewidth=3);
julia> scatter!((x, y), lab="Data");
julia> title!("Example usage of predint");
```
See also: [`confint`](@ref)
"""
function predint(mod::RegressionModel, xf::AbstractArray, p::Float64=0.95)
    # TODO: Ensure that `mod` is purely linear and of one variable
    # Convert `p` into a quantile-compatible format
    p = 0.5 + p/2;
    n = nrow(df); # Get the dimension of the dataset
    x_sym = mod.mf.f.rhs.sym; # Get the symbols
    y_sym = mod.mf.f.lhs.sym;
    X = mod.model.pp.X[:,2]; # Get the vector
    y = mod.model.rr.y;
    s2 = var(y); # Calculate the sample variance of y
    sx2 = var(x); # Calculate the sample variance of x
    xbar = mean(x); # Calculate the sample mean in x
    # Calculate the standard error with prediction intervals
    sse = sqrt.(s2 .* ((1 + (1/n)) .+ (xf .- xbar).^2 ./ ((n-1)*sx2)));
    # Get the t-value for the given p
    t_p = quantile(TDist(n), p);
    # Get the forecasted values
    pred = predict(mod, eval(:(DataFrame($x_sym = $xf))));
    # Calculate the deviance for the prediction interval
    dev = t_p*sse;
    # Calculate the interval
    interval = pred .+ dev*[-1 1];
    # Return all the appropriate values
    return pred, dev
end

"""
Calculate the studentized residuals of a model `mdl`

    rstudent(mdl)

# Arguments
  - `mdl::RegressionModel`- A regression model compatible with the GLM package

# Returns
  - `s_res::Array{float64}((n,))`- The studentized residual for each predicted value

# Examples
```jldoctest
julia> using DataFrames, GLM
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

end # module
