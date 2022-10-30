#####################################
### Functions for ICC calculation ###
#####################################

import numpy as np
from scipy.stats import f, chi2, t


def icc(g, e):
    """Calculates IntraClass Correlation

    Args:
        g (float): Sigma^2_g estimated variance for random effect(s)
        e (float): Sigma^2_e estimated variance for residual

    Returns:
        float: ICC
    """
    ICC = g / (e + g)
    print(f"*** ICC = {ICC:.4f} ***")
    return ICC


def confidence_interval_icc(MS_b, MS_w, df_b, df_w, n, alpha=0.05):
    """Calculate confidence intervals for ICC

    Args:
        MS_b (float): Mean Squares between (random effect in ANOVA table)
        MS_w (float): Mean Squares within (residual in ANOVA table)
        df_b (int): ANOVA table degrees of freedom, random effects
        df_w (int): ANOVA table degrees of freedom, residual
        n (float): Constant in ANOVA table, expected mean squares
        alpha (float, optional): CI, usually 95%. Defaults to 0.05.
    """
    F = MS_b / MS_w
    F_u = f.ppf(1 - alpha / 2, dfn=df_b, dfd=df_w)
    F_l = f.ppf(alpha / 2, dfn=df_b, dfd=df_w)
    lower = (F / F_u - 1) / (F / F_u + n - 1)
    upper = (F / F_l - 1) / (F / F_l + n - 1)
    print(f"*** Confidence interval: {lower:.4f} - {upper:.4f} ***")


#########################################
#### Sum of Variance components + CI ####
#########################################
from scipy.stats import chi2


def sum_of_var(Sigma_g, Sigma_e, MS_b, MS_w, cn, df_b, df_w, alpha=0.05):
    """Calculate total variance and confidence intervals

    Args:
        Sigma_g (float): VC for between group variance
        Sigma_e (float): VC for within group variance (residual)
        MS_b (float): Mean Squares between from ANOVA table
        MS_w (float): Mean Squares within (residual) from ANOVA table
        cn (float): Constant from expected mean squares column
        df_b (int): Degrees of freedom between groups from ANOVA table
        df_w (int): Degrees of freedom within groups from ANOVA table
        alpha (float, optional): Default 95% confidence for intervals. Defaults to 0.05.
    """
    ### Total variance estimate
    Sigma_t = Sigma_g + Sigma_e
    ### Variance of total variance estimate
    V_sigma_t = ((2 * MS_b**2) / (cn**2 * df_b)) + (
        (2 * (cn - 1) ** 2 * MS_w**2) / (cn**2 * df_w)
    )
    ### Satterthwaite degrees of freedom
    df_t = 2 * ((Sigma_t**2) / V_sigma_t)
    U_chi2 = chi2.ppf((1 - alpha / 2), df_t)
    L_chi2 = chi2.ppf(alpha / 2, df_t)
    LCL = (df_t * Sigma_t) / (U_chi2)
    UCL = (df_t * Sigma_t) / (L_chi2)
    print(f"*** Total Variability: {Sigma_t:.2f} ***")
    print(f"*** Confidence interval {LCL:.2f} - {UCL:.2f} ***")


###############################
#### Likelihood Ratio Test ####
###############################
from logging import warning


def likelihood_ratio_test(theta, theta_0, df):
    """Do a likelihood ratio test

    Args:
        theta (float): Log-likelihood from extended model
        theta_0 (float): Log-likelihood from reduced (null) model
        df (int): Degrees of freedom for test (difference in params)
    """
    LRT = theta_0 - theta
    p = 1 - chi2.cdf(LRT, df)
    print(
        f"*** P-value from Chi-Square: {p:.4f} *** \n *** If below 0.05, reject null hypothesis ***"
    )
    if p < 1e-4:
        warning(" P-value extremely low!")


#############################################
#### Minimize variance for time profiles ####
#############################################


def minimze_variability(tau_0, tau_1, off_diag, s2_r):
    """Minimize variablity for linear time profiles
    in subject-specific models

    Args:
        tau_0 (float): variance of intercept
        tau_1 (float): variance of slope
        off_diag (float): value in the off diagonal (NOT EQUAL TO rho)
        s2_r (float): estimated residual variance
    """
    rho = off_diag / (np.sqrt(tau_0) * np.sqrt(tau_1))
    min_var = (1 - rho**2) * tau_0 + s2_r
    min_t = -rho * np.sqrt(tau_0) / np.sqrt(tau_1)
    print(
        f"*** Minimum variance: {min_var:.3f} *** \n*** Minimum variance achieved at time point {min_t:.3f} ***"
    )
