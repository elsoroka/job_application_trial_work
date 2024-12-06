import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def __():
    # this notebook can be run either as a standalone python file
    # or using marimo
    # `pip install marimo`
    # then `marimo run notebook.py`

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy
    return mo, mpl, np, pd, plt, scipy


@app.cell
def __(mo):
    mo.md(
        r"""
        Implement and code the optimal strategy with Linear Impact and visualize the Sharpe Ratio
        plots in Section 6.2.

        Assumptions:

        * we use the same $\beta$, $\lambda$, $\sigma$, $\phi$, $\alpha$, as in the paper.
        """
    )
    return


@app.cell
def __(np, scipy):
    BETA_N = 2.0
    LAMBDA_N = 0.0035
    sigma = 0.02 # this is from page 9, section 6.1
    phi = 0.139 # also from page 9, section 6.1
    alpha = 1.67E-4 #sigma * 3.0 / np.sqrt(256.0) # section 6.1

    from collections import namedtuple
    Params = namedtuple('Params', ['beta', 'lm', 'sigma', 'phi', 'alpha', 'gamma', 'p', 'beta_n', 'lm_n', 'gamma_n'])

    def r_squared(p, beta, lm, gm):
        tmp = (p.alpha*p.sigma)**2 / ( gm**2 * p.sigma**4 )
        term1 = np.sqrt( 1.0 + 2.0 * lm * beta / (gm * p.sigma**2) )
        tmp *= (beta / p.phi + 1)**2.0 * (beta/p.phi * term1 + 1.0) / ( term1 * (term1 + beta/p.phi)**3.0 )
        return tmp

    def init_params(risk_constraint, beta=2.0, lm=0.0035, sigma=0.02, phi=0.139, p=0.5):
        alpha = 1.67E-4#sigma * 3.0 / np.sqrt(256.0) # section 6.1
        params = Params(beta, lm, sigma, phi, alpha, 0.0, p, BETA_N, LAMBDA_N, 0.0)
        # now we find gamma which is defined as the risk aversion parameter where R(Q*) is at the desired level

        def rootfind1(gm):
            return r_squared(params, params.beta, params.lm, gm) - risk_constraint**2

        # the rootfinding interval is chosen to avoid 0.0, where R is infinity
        # the choice of ending at 10.0 is arbitrary
        sol1 = scipy.optimize.root_scalar(rootfind1, method='bisect', bracket=[1e-16, 10.0])

        def rootfind2(gm):
            return r_squared(params, params.beta_n, params.lm_n, gm) - risk_constraint**2

        sol2 = scipy.optimize.root_scalar(rootfind2, method='bisect', bracket=[1e-16, 10.0])

        params = Params(beta, lm, sigma, phi, alpha, sol1.root, p, BETA_N, LAMBDA_N, sol2.root)

        return params


    def compute_pnl(params):
        gm = params.gamma
        sg = params.sigma
        alpha = params.alpha
        beta = params.beta
        lm = params.lm
        phi = params.phi
        p = params.p

        # eq. 5.2
        zeta = np.sqrt(gm * sg**2 * (gm * sg**2 + 2.0 * lm * beta))

        # eq. 5.4
        #denom = (beta*gm*sg**2 - zeta*phi) * (beta*gm*sg**2 + zeta*phi)**3
        #C1 = ( (alpha * gm * sg**2)**2 * (beta - phi) * (beta + phi)**3 ) / ( denom )
        # eq. 5.5
        #C2 = ( 2.0*alpha**2*gm**2*lm*beta**2*sg**4*phi*(beta + phi)**2 ) / ( zeta*denom )

        # for the linear feedback, equation 5.6 gives PnL
        term1 = ( alpha**2*gm*sg**2 * (beta+phi)**2 ) / ( (beta*gm*sg**2 + zeta*phi)**2 )
        # here we have n=1

        beta_n = params.beta_n
        lm_n = params.lm_n

        term2 = ( 2.0*alpha**2*gm**2*sg**4*phi*(beta+phi)**2 * (beta**2*zeta + beta_n*phi*zeta + gm*beta*sg**2*(beta_n + phi)) )
        term2 /= ( zeta*(beta_n + phi) * (gm*beta*sg**2 + beta_n*zeta) * (phi*zeta + gm*beta*sg**2)**3 )
        term2 = term2 ** ((p+1)/2)

        term2 *= scipy.special.gamma(p/2.0) * p*beta_n*lm_n/(2.0*np.sqrt(np.pi))

        pnl = term1 - term2
        return pnl
    return (
        BETA_N,
        LAMBDA_N,
        Params,
        alpha,
        compute_pnl,
        init_params,
        namedtuple,
        phi,
        r_squared,
        sigma,
    )


@app.cell
def __(BETA_N, LAMBDA_N, compute_pnl, init_params, np, r_squared):
    # for plotting
    def ratio(lm=LAMBDA_N, beta=BETA_N, risk=0.5):
        p = init_params(risk, lm=lm, beta=beta)
        pnl = compute_pnl(p)
        r2 = r_squared(p, p.beta, lm, p.gamma)
        return pnl/np.sqrt(r2)
    return (ratio,)


@app.cell
def __(np, plt, ratio):
    lm_range = np.arange(0.005,0.04,0.001)
    beta_range = np.arange(2.0, 7.0, 0.1)

    fig, axs = plt.subplots(3,2)
    fig.suptitle("Task 2")
    for (i,r) in enumerate([0.5, 1.3, 5.3]):
        axs[i][1].set_xlabel(f"lambda, risk={r}")
        axs[i][0].set_xlabel(f"beta, risk={r}")

    for (i,r) in enumerate([0.5, 1.3, 5.3]):

        axs[i][1].plot(lm_range, [ratio(lm=lm_i, risk=r) for lm_i in lm_range])
        axs[i][0].plot(beta_range, [ratio(beta=beta_i, risk=r) for beta_i in beta_range])

    plt.tight_layout()
    plt.savefig("task2.pdf")
    plt.show()
    return axs, beta_range, fig, i, lm_range, r


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
