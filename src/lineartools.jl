
function predint(mod::StatsBase.RegressionModel, xf::AbstractArray, p::Float64)
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

function studentized_residuals(mdl::StatsBase.RegressionModel)
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