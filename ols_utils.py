import numpy as np


def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Ordinary Least Squares formulas:

    Given:
        X : n x p design matrix
        y : n x 1 response vector

    Estimate coefficients:
        beta = (X^T X)^(-1) X^T y

    Fitted values:
        yhat = X beta

    Residuals:
        resid = y - yhat

    Residual sum of squares:
        SS_res = resid^T resid

    Total sum of squares:
        SS_tot = (y - mean(y))^T (y - mean(y))

    R-squared:
        R^2 = 1 - SS_res / SS_tot

    Adjusted R-squared:
        adj_R^2 = 1 - (1 - R^2) * (n - 1) / (n - p)

    Variance of beta:
        Var(beta) = sigma^2 * (X^T X)^(-1)
        where sigma^2 = SS_res / (n - p)

    Standard error:
        se_beta = sqrt(diag(Var(beta)))
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