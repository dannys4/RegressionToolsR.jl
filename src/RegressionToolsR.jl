module RegressionToolsR

import GLM
import DataFrames
import LinearAlgebra
import Distributions
import StatsBase

include("lineartools.jl");

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
export predint = prediction_interval(mod, xf, p=0.95);

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
export rstudent(mdl) = studentized_residuals(mdl);

end # module
