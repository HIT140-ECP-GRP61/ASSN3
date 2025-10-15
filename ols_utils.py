import numpy as np

def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Ordinary Least Squares regression using linear algebra.
    Returns:
        beta: Coefficient estimates
        se_beta: Standard errors
        yhat: Fitted values
        resid: Residuals
        r2: R-squared
        adj_r2: Adjusted R-squared
        dof: Degrees of freedom
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat

    n, p = X.shape
    dof = max(n - p, 1)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) @ (y - y.mean())))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1 - r2) * (n - 1) / dof

    sigma2 = ss_res / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.clip(np.diag(var_beta), 0.0, np.inf))

    return beta, se_beta, yhat, resid, r2, adj_r2, dof