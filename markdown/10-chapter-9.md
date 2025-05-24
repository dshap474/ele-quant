**Chapter 9**
**Portfolio Management: The Basics**

**The Questions**

1.  Why is Mean-Variance Optimization (MVO) widely used in portfolio construction, and what are its main assumptions and limitations?
2.  How does the structure of the utility function impact investment decisions, specifically in terms of expected return and risk?
3.  How do we add factor risk to MVO? What does it mean to trade in factor space and in idio space?
4.  How do we add a factor to the model?
5.  How does the Information Ratio relate to an investor’s skill (Information Coefficient) and the diversification of a portfolio?
6.  Should a centralized approach to portfolio management be used over a decentralized one, and under what conditions are they equivalent?

This chapter is devoted to the basics of portfolio construction. The common theme throughout the chapter is that we limit ourselves to a single-period optimization setting. This is a chapter for hedgehogs, not for foxes: we set a narrow playing field, but dig a deep hole. The chapter requires knowledge of basic results from optimization theory.[1]

---

**9.1 Why Mean-Variance Optimization?**

Investors have objectives, information, and constraints. Besides this generic statement, there is not much in common among them. A large fraction of investment professionals cannot—and would not—articulate a clear objective function; their constraints are sometimes ad hoc, vague, or inconsistently enforced. Neither George Soros nor Warren Buffett, nor others among the most successful investors in recent history, would have used the tools of this chapter at any point in their career. At the other extreme, academics have applied optimization techniques for portfolio construction. In this book I use relevance to applications as a guiding principle. In the vast majority of applications, the optimization formulations are single-period. This is explainable by a combination of the following:
*   Interpretability. Multi-period optimization problems are vastly more complex to formulate and, once solved, their solutions are also harder to interpret.
*   Data availability. The amount of data needed for multi-period optimization is larger and more difficult to estimate.
*   Computational tractability. Single-period optimization problems are solvable by commercial solvers in a matter of seconds.
*   Usefulness. Multi-period optimization problems are all about the short term, partly because they heavily discount future information, partly because they do not know how to quantify information uncertainty and rate of change.

The objective function $E[u(W)]$ is a function of the portfolio weights via the end-of-period wealth $W$. The objective function $E[u(W)]$ is a utility function, taking different values under different realizations of the future. The expected value of the utility function gives the investor the ex-ante value of the utility. She would be taking by investing in a portfolio. We assume that the investor has initial wealth $W_0$, that she knows the distribution of the random vector $r$, and that she solves the problem
(9.1)
$$ \max_{\mathbf{w}} E[u(W_0 + \mathbf{w}^T \mathbf{r})] $$
The choice of $u(W)$ is not obvious. Common properties of $u'(W) > 0$ that it must be monotonically increasing (more wealth is better than less) and concave (corresponding to risk aversion, and to decreasing value of a marginal dollar as a function of wealth). One approach, followed by Markowitz (1952), is to consider a polynomial approximation of the utility function. If we assume that the investor’s utility is well approximated by a second-order Taylor expansion, then
$$ E[u(W_0 + \mathbf{w}^T \mathbf{r})] \approx u(W_0) + u'(W_0) E[\mathbf{w}^T \mathbf{r}] + \frac{1}{2} u''(W_0) E[(\mathbf{w}^T \mathbf{r})^2] $$
$$ E[V_0(W_0 + \mathbf{w}^T \mathbf{r})] = V_0(W_0) + V_0'(W_0) E[\mathbf{w}^T \mathbf{r}] + \frac{V_0''(W_0)}{2} (\mathbf{w}^T \Omega_r \mathbf{w} + (E[\mathbf{w}^T \mathbf{r}])^2) $$
$$ = V_0(W_0) + V_0'(W_0) E[\mathbf{w}^T \mathbf{r}] + \frac{V_0''(W_0)}{2} \mathbf{w}^T \Omega_r \mathbf{w} $$
We maximize a concave quadratic objective function which is the weighted sum of expected return and variance; hence the name Mean-Variance Optimization (MVO) (De Finetti, 1940; Markowitz, 1952)
$$ E[V_0(W_0 + \mathbf{w}^T \mathbf{r})] - V_0(W_0) = V_0'(W_0) \mathbf{w}^T \mu_r - \frac{\rho}{2} \mathbf{w}^T \Omega_r \mathbf{w} $$
$$ \rho = - \frac{V_0''(W_0)}{V_0'(W_0)} $$
$\rho > 0$ is called the coefficient of absolute risk aversion (CARA). The higher the $\rho$, the more risk-averse the investor is.
As examples, consider an objective function of the form
$$ V(x) = -\exp(-ax) $$
The CARA for this function is constant $\rho = a$. It is independent of the wealth $W_0$ of the investor and so are her allocation decisions. The optimization problem is
$$ \max_{\mathbf{w}} \mathbf{w}^T \mu_r - \frac{a}{2} \mathbf{w}^T \Omega_r \mathbf{w} $$
Alternatively, consider the objective function $V(x) = \log(x)$. This function is associated to the fully criterion for investing. It has unique properties which warrant a dedicated chapter. Here, let us consider its implications for approximate portfolio optimization. The CARA is $\rho_W = 1/W_0$, so that we solve
$$ \max_{\mathbf{w}} \mathbf{w}^T \mu_r - \frac{1}{2W_0} \mathbf{w}^T \Omega_r \mathbf{w} $$
The wealthier the investor is, the more risk-seeking she becomes.
We have shown that a quadratic utility function implies a mean-variance optimization problem for the investor. This result is standard. Less known is the converse: an investor selects an investment on the basis of mean and variance if and only if her utility function is quadratic (Chamberlain, 1983; Owen and Rabinovitch, 1983). Viewed in the context of axiomatic decision theory, portfolio MVO is not satisfactory, because a quadratic utility implies that investors are satiated, and have even a dislike of wealth beyond a certain threshold. As a local approximation, however, the quadratic approximation is useful. Indeed, the results derived from MVO are robust to the realm of what is plausible, as seen in Chapter 3. A portfolio manager settled a long discussion on the topic with the laconic statement that “the first two moments should be enough for everybody”.

---

**9.2 Mean-Variance Optimal Portfolios**

A factor model gives us an asset-asset covariance matrix $\Omega_r \in \mathbb{R}^{N \times N}$. Given this information, it is straightforward to compute the variance portfolio, as set out in Section 3.2, for risk decomposition. The other essential input to the optimization problem is a vector $\alpha \in \mathbb{R}^N$ of expected returns, over the same interval at which we have a volatility forecast. The simplest optimization problem is to maximize expected PnL, subject to a constraint on the maximum tolerable volatility, denoted by $\sigma^2$. The problem can be stated as
(9.2)
$$ \max_{\mathbf{w}} \alpha^T \mathbf{w} $$
$$ \text{s.t. } \mathbf{w}^T \Omega_r \mathbf{w} \le \sigma^2 $$
One of the most important metrics used for the evaluation of strategies is the Sharpe Ratio. If we have covariance matrix and expected returns, we can formulate the Sharpe Ratio optimization thus:
$$ \max_{\mathbf{w}} \frac{\alpha^T \mathbf{w}}{\sqrt{\mathbf{w}^T \Omega_r \mathbf{w}}} $$
This optimization, however, is indefinite because the objective function $\text{SR}(\mathbf{w})$ is independent of the portfolio size, i.e., homogeneous of degree 0: $\text{SR}(c\mathbf{w}) = \text{SR}(\mathbf{w})$ for $c > 0$. We can address this issue by adding the constraint that the portfolio has unit volatility. The additional condition the denominator is always binding if there is a portfolio $\mathbf{w}^*$ such that $\alpha^T \mathbf{w}^* > 0$.
$$ \max_{\mathbf{w}} \frac{\alpha^T \mathbf{w}}{\sqrt{\mathbf{w}^T \Omega_r \mathbf{w}}} $$
$$ \text{s.t. } \sqrt{\mathbf{w}^T \Omega_r \mathbf{w}} \le \sigma $$
equivalent to $\max_{\mathbf{w}} \frac{\alpha^T \mathbf{w}}{\sigma}$
$$ \text{s.t. } \mathbf{w}^T \Omega_r \mathbf{w} \le \sigma^2 $$
equivalent to $\max_{\mathbf{w}} \alpha^T \mathbf{w}$
$$ \text{s.t. } \mathbf{w}^T \Omega_r \mathbf{w} \le \sigma^2 $$
which is Optimization Problem (9.2). The First-Order Necessary Conditions (FONCs) for this problem are
$$ \nabla_{\mathbf{w}} (\alpha^T \mathbf{w} - \lambda \mathbf{w}^T \Omega_r \mathbf{w}) = \alpha - 2\lambda \Omega_r \mathbf{w} = 0 $$
$$ \mathbf{w}^T \Omega_r \mathbf{w} \le \sigma^2 $$
$$ \lambda \ge 0 $$
$$ \lambda(\mathbf{w}^T \Omega_r \mathbf{w} - \sigma^2) = 0 $$
The solution to these equations is
(9.3)
$$ \mathbf{w}^* = \frac{\sigma}{\sqrt{\alpha^T \Omega_r^{-1} \alpha}} \Omega_r^{-1} \alpha $$
(9.4)
$$ \lambda^* = \frac{\sqrt{\alpha^T \Omega_r^{-1} \alpha}}{2\sigma} $$
The expected return and the Sharpe Ratio of the portfolio are
(9.5)
$$ E(\mathbf{r}^T \mathbf{w}^*) = \sigma \sqrt{\alpha^T \Omega_r^{-1} \alpha} $$
(9.6)
$$ \text{SR}^* = \sqrt{\alpha^T \Omega_r^{-1} \alpha} $$
A way to interpret (and derive quickly) the solution is to recall that the optimal portfolio is proportional to $\Omega_r^{-1} \alpha$, and then to find the proportionality factor so that the variance constraint is met. The optimization problem is proportional to the volatility target $\sigma$. The larger the budget, the larger the return. However, the Sharpe Ratio is independent of the magnitude of the alpha vector (it is homogeneous of degree zero in alpha), replacing $\alpha$ with $c\alpha$ gives the same solution. This is interesting.

**Insight 9.1: Misallocation of alpha size is not catastrophic**
If you have a volatility constraint, a good volatility model, and your relative alphas are accurate, then the error in the absolute size of the alphas does not matter.

The parameter $\lambda^*$ also merits special consideration. It is the shadow price (or Lagrange multiplier) of the volatility constraint. If we increase the variance constraint by one unit, the expected return increases by $\lambda^*$. In other terms, the shadow price of the variance constraint is the derivative of the objective function with respect to the variance. This relationship is not very useful in this specific case; it will come handy in other cases.
In its simplicity, the solution contains the essential data of the problem: the inverse of the covariance matrix (also called the precision matrix), and the vector of expected returns. In the next few pages we will interpret, extend, and use this simple functional form. And finally, as we start believing it is useful, we will caution you against its unconditional use. Like all good things in life, MVO is at its most pleasant when it is accompanied by precautionary measures.

**Insight 9.2: MVO from ex-post Sharpe and correlations**
There is yet another formulation that is equivalent to the previous ones. Oftentimes, we think of portfolio positions not in terms of NAVs, but of volatility. We do not invest $10M in AAPL. The annualized volatility of AAPL is 20%, and therefore we have a $2M volatility position in the stock. This convention is useful when we want to manage the risk of the portfolio over the course of a year. Now, we can express the Sharpe-optimal portfolio in terms of volatility in the following way. Let the stock volatilities be $D_v = \text{diag}(\sigma_1, \dots, \sigma_N)$, and define $V_s$ a diagonal matrix whose entries are the volatilities to be chosen. Then, the optimal dollar-volatility allocation is $\mathbf{v} = D_v \mathbf{w}$. Now rewrite the solution to the MVO problem:
(9.7)
$$ \mathbf{w}^* = \frac{1}{2\lambda^*} (\mathbf{V} C \mathbf{V})^{-1} \alpha $$
$$ \mathbf{V} \mathbf{w}^* = \frac{1}{2\lambda^*} C^{-1} \mathbf{V}^{-1} \alpha $$
$$ \mathbf{v}^* = \frac{1}{2\lambda^*} C^{-1} \mathbf{s} $$
(9.8) $\text{SR}^* = \sqrt{\mathbf{s}^T C^{-1} \mathbf{s}}$ is the vector of optimal dollar volatilities. $\mathbf{s}$ is the vector of ex-post Sharpe Ratios. Therefore, the optimal dollar volatilities are proportional to the Sharpe Ratios, multiplied by the inverse of the correlation matrix. This is interesting, because dollar volatilities, rather than NAVs, are more intuitive quantities than constraints, and correlations.
First of all, we can derive the same solution when we solve an unconstrained problem:
(9.9)
$$ \max_{\mathbf{w}} \alpha^T \mathbf{w} - \frac{\lambda}{2} \mathbf{w}^T \Omega_r \mathbf{w} $$
s.t. $\mathbf{w} \in \mathbb{R}^N$
We have added the constraint to the objective function in the form of a penalty term; the informal term for this operation is giving out the constraint. The objective function is concave, and the solution is given by
$$ \mathbf{w}^* = \frac{1}{\lambda} \Omega_r^{-1} \alpha $$
which gives the same solution as the vol-constrained problem when
$$ \lambda = \frac{\sqrt{\alpha^T \Omega_r^{-1} \alpha}}{2\sigma} $$
The larger the volatility budget, the smaller the penalty coefficient.
Notice that this penalty value is the same as the shadow price in the previous formulation. This is not a coincidence. We obtain the same solution when we price out the constraint and we give the variance a unit price equal to the shadow price of that constraint.
A third equivalent formulation is the one where we minimize volatility, subject to a return constraint:
(9.10) $\min_{\mathbf{w}} \mathbf{w}^T \Omega_r \mathbf{w}$
(9.11) s.t. $\alpha^T \mathbf{w} \ge \mu$
The solution is
$$ \mathbf{w}^* = \frac{\mu}{\alpha^T \Omega_r^{-1} \alpha} \Omega_r^{-1} \alpha $$

**Insight 9.3: Asset correlations, dispersion, and limits to performance**
From Equation (9.8), when assets are uncorrelated, the optimal dollar-vol allocation is proportional to the asset Sharpe Ratios and the correlation matrix is the identity matrix $C=I_N$. Which is not true when the assets are correlated. If two assets have a positive correlation $\rho > 0$, then the optimal volatility allocated to asset $j$ in the two-asset case (and only) is proportional to the excess Sharpe Ratio of the asset, compared to the excess Sharpe Ratio of the asset, compared to the other one:
$$ v_j \propto S_j - \rho S_i $$
where we define $S_j = \text{SR}_j / \sqrt{1-\rho^2}$. The optimal Sharpe Ratio is
$$ \text{SR}^* = \sqrt{\frac{1}{1-\rho^2}} \sqrt{\text{var}(S) + \frac{n}{1+(n-1)\rho} E^2(S)} $$
with
$$ \text{var}(S) = E(S^2) - E^2(S) $$
If all assets have the same Sharpe Ratio $s$, then
$$ \text{SR}^* = \sqrt{\frac{ns^2}{1+(n-1)\rho}} $$
Finally, in the many-asset limit, we have $\text{SR}^* \approx s/\sqrt{\rho}$.
Summing up the results above:
1.  If there is no dispersion in Sharpe Ratios, then the Sharpe Ratio approaches an upper bound $s/\sqrt{\rho}$.
2.  If there is dispersion in Sharpe Ratios, then the Sharpe Ratio is still proportional to $\sqrt{N}$, and to the dispersion, measured as the cross-sectional standard deviation of the assets’ Sharpe Ratios.
3.  To prove this directly, verify that the inverse of the correlation matrix is
    $$ C^{-1} = \frac{1}{1-\rho} \left( I_N - \frac{\rho}{1+(n-1)\rho} \mathbf{e} \mathbf{e}^T \right) $$

**Insight 9.4: Reading the entries of the precision matrix**
Is there a way to interpret further the relationship $\mathbf{w}^* \propto \Omega_r^{-1} \alpha$? The optimal position of asset $i$ is a weighted sum of alphas. The $P_{ij} = (\Omega_r^{-1})_{ij}$ are proportional to minus the partial correlation of the returns of $i$ and $j$, after controlling for the effect of all other assets. The interpretation of partial correlation is that it captures collinearity between two random variables, after removing the collinearity of these variables with a set of controlling variables. In practice, one follows this procedure: regress the returns of asset $i$ and $j$ on the returns of the other assets, and compute the correlation between the residuals from the two regressions, which we denote $A_{ij}$. The formula for the optimal portfolio is
$$ \mathbf{w}_i^* \propto (\Omega_r^{-1})_{ii} \alpha_i - \sum_{j \ne i} (\Omega_r^{-1})_{ij} \alpha_j $$
The diagonal terms of the precision are always positive. The interpretation of this, rather counterintuitive formula is that, whenever the returns of two assets are positively correlated after removing the joint effect of correlations with other variables, the size of the portfolio is reduced. Because the collinearity makes the alpha common to both assets and $j$.

---

**9.3 Trading in Factor Space**

**9.3.1 Factor-Mimicking Portfolios**

We have a factor model, and we estimate the expected factor returns $\hat{\lambda}$. Say that we want to generate a portfolio which has unit positive exposure to one of the factors, say factor $k$, and zero exposure to all other factors. This portfolio is called the Factor-Mimicking Portfolio (FMP). We should be clear that the returns of the Factor-Mimicking Portfolio should be as close as possible to those of the factor. The variance of the difference of the returns should be minimized. A portfolio $\mathbf{w}$ has an associated factor exposure $\mathbf{b}_k = \mathbf{B}^T \mathbf{w}$. Its returns are $\mathbf{w}^T \mathbf{r} = \mathbf{w}^T \alpha + \mathbf{b}^T \mathbf{f} + \mathbf{w}^T \mathbf{\epsilon}$. The tracking variance between $\hat{f}_k$ and $\mathbf{w}^T \mathbf{r}$ is $E[(\hat{f}_k - (b_k - 1)f_k - \sum_{j \ne k} b_j f_j + \mathbf{w}^T \mathbf{\epsilon})^2]$. This is minimized when $b_k=1, b_j=0, j \ne k$, and the portfolio’s idiosyncratic variance is minimized. The optimization formulation is
$$ \min_{\mathbf{w}} \mathbf{w}^T \Omega_\epsilon \mathbf{w} $$
$$ \text{s.t. } \mathbf{B}^T \mathbf{w} = \mathbf{e}_k $$
The solution is
$$ \mathbf{v}_k = \Omega_\epsilon^{-1} \mathbf{B} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{e}_k $$
The matrix whose column vectors are the FMPs is $\mathbf{P} \in \mathbb{R}^{N \times K}$ (“portfolio for factors”).
(9.12)
$$ \mathbf{P} := [\mathbf{v}_1 \quad \mathbf{v}_2 \quad \dots \quad \mathbf{v}_K] $$
$$ = \Omega_\epsilon^{-1} \mathbf{B} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} [\mathbf{e}_1 \quad \mathbf{e}_2 \quad \dots \quad \mathbf{e}_K] $$
$$ = \Omega_\epsilon^{-1} \mathbf{B} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} $$
We now have portfolios as tradable instruments. The expected return of a factor portfolio is
$$ (\alpha^T + \lambda^T \mathbf{B}^T) \mathbf{P} \mathbf{e}_k = \lambda_k $$
In practice, we:
1.  generate the FMPs compositions over time;
2.  compute their per-period PnL;
3.  compute their empirical average PnL;
4.  apply modifications to the expected PnL, such as penalties;
5.  now prove that FMPs emerge naturally from certain assumptions about returns and FMP composition, which we use in the Appendix, Section 9.7.2. We prove that if alpha spanned and zero if the idiosyncratic variance is small, then the MVO problem reduces to one in which we only trade FMPs. First, we solve the new-dimensional optimization problem
    $$ \max_{\mathbf{u}} \lambda^T \mathbf{u} - \frac{1}{2\gamma} \mathbf{u}^T \Omega_f \mathbf{u} $$
    $$ \text{s.t. } \mathbf{u} \in \mathbb{R}^m $$
    Say that the solution is $\mathbf{u}^*$. The optimal portfolio is the weighted sum of the FMPs: $\mathbf{w}^* = \mathbf{P} \mathbf{u}^*$. A few remarks on the appeal and limitations of the previous result:
    *   In factor space, the dimensionality of the problem collapses, but we still have an MVO problem, which is usually more interpretable. The problem is still an MVO problem, but in factor space. We can solve this, but to do so is possible only when we proceed in Chapter 7.
    *   FMPs make their appearance as the necessary synthetic instruments for trading in factor space. In synthesis, if we perform MVO and factor risk is sufficiently low (having low-idiosyncratic risk), then we necessarily trade FMPs. This means that the optimal portfolio is a linear map in mind that FMPs are associated to the loadings matrix $\mathbf{B}$; and there are many loadings matrices resulting in equivalent factor models (see Section 4.4). Think of FMPs as a vector basis in a finite-dimensional space. There are infinitely many such bases, and they don’t need to be orthogonal. This factor-based viewpoint is currently stifled.
    *   The assumption of small factor idiosyncratic variance can be expressed as
        $$ \| (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \| < 1 $$
    The matrix inside the norm appears repeatedly in this book—in an ideal world we would like it to be small, but in the real world it is not. The observed values could be as large as a factor of 100. This is the main challenge of the portfolio construction, and not solve for the ideal case.
    *   In conjunction with the previous point, we have also ignored execution costs. They should not be ignored in applications. Chapter 10 is devoted to this subject.

---

**9.3.2 Adding, Estimating, and Trading a New Factor**

Let us consider an instructive example, and we work through the individual steps in a way, all the steps are implicitly contained in the theory developed so far. The starting point is a factor model with $m$ factors, with parameters $\mathbf{B}_m, \Omega_f, \alpha_m$. We assume that they are constant through time for notational simplicity, extending the example to time-varying parameters is straightforward. The factors have expected returns $\lambda_m$. We are exploring a new asset characteristic vector $\mathbf{a} \in \mathbb{R}^N$, and we would like to add it as a factor. We could add it to the existing model without orthogonalizing it to the existing model by pushing out the model.
1.  Orthogonalization. First we orthogonalize the new factor to the existing factors. The orthogonalized factor is given by standard linear regression formulas:
    $$ \mathbf{b}_{m+1} = (I_N - \mathbf{B}_m (\mathbf{B}_m^T \mathbf{B}_m)^{-1} \mathbf{B}_m^T) \mathbf{a} $$
2.  Estimation. Next, we regress in every period the residual returns from the existing model against the orthogonalized factor by using the Frisch-Waugh-Lovell Theorem:[8]
    (9.13)
    $$ \hat{\epsilon}_t = \mathbf{b}_{m+1} f_{m+1,t} + \mathbf{\epsilon}_{m+1,t} $$
    (9.14)
    $$ \hat{f}_{m+1,t} = \frac{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \hat{\epsilon}_t}{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}} $$
    (9.15)
    $$ \hat{\lambda}_{m+1} = \frac{1}{T} \sum_{t=1}^T \hat{f}_{m+1,t} $$
    (9.16)
    $$ \hat{\sigma}_{m+1}^2 = \frac{1}{T} \sum_{t=1}^T (\hat{f}_{m+1,t} - \hat{\lambda}_{m+1})^2 $$
    (9.17)
    $$ \hat{\Omega}_{\epsilon, m+1} = \hat{\Omega}_{\epsilon,m} - \frac{\hat{\sigma}_{m+1}^2}{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}} \mathbf{b}_{m+1} \mathbf{b}_{m+1}^T $$
    In the Appendix (Theorem 9.2), we show that the approximate variance of the new factor is
    $$ \sigma_{m+1}^2 = (\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1})^{-1} $$
3.  Risk updating. We show in Section 9.7.2 that the approximate factor covariance matrix including the new factor is given by
    $$ \hat{\Omega}_f \approx \begin{bmatrix} \hat{\Omega}_{f,m} & \mathbf{0} \\ \mathbf{0}^T & (\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1})^{-1} \end{bmatrix} $$
    This result holds only for constant parameters and well-diversified factor portfolios. For time-varying models, we need to resort to numerical estimation for the factor covariance matrix. The analytical results provide a useful approximation.
4.  Trading. The FMP of the new factor is
    $$ \mathbf{v}_{m+1,t} = \frac{\Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}}{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}} $$
    This follows from the definition of FMPs, Equation (9.12). A faster route is via the factor return estimation above. The factor return estimate is the same whether we regress using residual returns or total returns:[9]
    $$ \hat{f}_{m+1,t} = \frac{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{r}_t}{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}} $$
    $$ = \mathbf{v}_{m+1,t}^T \mathbf{r}_t $$
    Let $\tilde{\lambda} = (\lambda_m, \lambda_{m+1})$. Finally, we solve the optimization problem
    $$ \max_{\mathbf{u}} \tilde{\lambda}^T \mathbf{u} - \frac{1}{2\gamma} \mathbf{u}^T \tilde{\Omega}_f \mathbf{u} $$
    $$ \text{s.t. } \mathbf{u} \in \mathbb{R}^{m+1} $$
    whose solution is simple, because of the block structure of the factor covariance matrix
    $$ \mathbf{u}^* = \gamma \begin{bmatrix} (\hat{\Omega}_{f,m})^{-1} \lambda_m \\ (\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}) \lambda_{m+1} \end{bmatrix} $$
    So, our investment in the pre-existing factors is unchanged, but we add a position in the new factor, proportional to the expected factor return divided by its variance.

**Procedure 9.1: Adding a new factor to a model and trading it**
1.  Inputs: a factor model ($\mathbf{B}_m, \Omega_f, \Omega_{\epsilon,m}, \alpha_m$), with expected factor returns $\lambda_m$; raw loadings for the new factor $\mathbf{a}$.
2.  Orthogonalize the factor:
    $$ \mathbf{b}_{m+1,t} = (I_N - \mathbf{B}_m (\mathbf{B}_m^T \mathbf{B}_m)^{-1} \mathbf{B}_m^T) \mathbf{a}_t $$
3.  Compute the FMP:
    $$ \mathbf{v}_{m+1,t} = \frac{\Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1,t}}{\mathbf{b}_{m+1,t}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1,t}} $$
    $$ \hat{\lambda}_{m+1} = \frac{1}{T} \sum_{t=1}^T \mathbf{v}_{m+1,t}^T \mathbf{r}_t $$
4.  Compute new factor covariance matrix:
    $$ \hat{\Omega}_f \approx \begin{bmatrix} \hat{\Omega}_{f,m} & \mathbf{0} \\ \mathbf{0}^T & (\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1})^{-1} \end{bmatrix} $$
5.  Compute new weights for FMPs:
    $$ \mathbf{u}^* = \gamma \begin{bmatrix} (\hat{\Omega}_{f,m})^{-1} \lambda_m \\ (\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}) \lambda_{m+1} \end{bmatrix} $$
and trade portfolios.
$$ \mathbf{w}_t = [\mathbf{v}_{1,t} \quad \dots \quad \mathbf{v}_{m,t} \quad \mathbf{v}_{m+1,t}] \mathbf{u}_t^* $$

**9.3.3 Factor Portfolios from Sorts?**

A very popular way to form factor portfolios is to sort securities by a given characteristic, and then form a portfolio in which the long positions are the top $q$-quantile positions (say, the top 10% positions) and the shorts are the bottom $q$-quantile positions. These portfolios (FMPs) in which each position is identical, so that the portfolio is dollar-neutral. For example, to construct a value-minus-growth portfolio for a U.S. investment universe of 3000 stocks, and go long $q=1000$ the 1000 stocks with the highest value of the characteristic, and short $q=1000$ the 1000 stocks with the lowest value of the ratio. This approach originates with the paper by Fama and French (1993), who construct factor portfolios based on size and value. The main motivation for this approach is simplicity, by estimating portfolios instead. The resulting portfolios are sometimes called portfolio sorts or characteristic portfolios (CRs; Daniel et al. (2020)). They are widely used by practitioners to convert a metric that describes a potential mispricing into an investable portfolio. What are the drawbacks of this approach? There are at least four:
1.  The resulting portfolios have unit exposure to one factor. As a result, they may be heavily exposed to the characteristics of other factors, as well as to some other sources of systematic risk and return. The unwanted exposures may both increase the overall portfolio risk and reduce the return of the portfolio.
2.  The securities in the characteristic portfolio are equally weighted. Intuitively, one may want securities that have lower idiosyncratic risk to have a higher weight.
3.  Being equal-weighted, the sizes in the characteristic portfolio do not reflect the magnitude of the characteristic. For example, if asset A has a value of the characteristic of 10, and asset B has a value of 0.1, and they are both in the top $q$-quantile of the characteristic, they receive the same weight. This is the same as dichotomizing the characteristic. Dichotomization of data is usually a poor modeling strategy (e.g., Harrell (2015)).
4.  Over time, due to business change, and certain securities become illiquid/delisted to new securities. This makes trading more expensive and requires some adjustment over the naive weights.

**FAQ 9.1: What about factor portfolios from sorts?**
Portfolios from sorts (or characteristic portfolios) are dollar neutral portfolios which consist of equal-weighted long positions in securities having the highest values of a certain characteristic, and equal-weighted short positions for an equal number of securities with the lowest value of a characteristic. They hold the intuitive appeal of being a simple representation of a dollar-neutral portfolio. However, they have major drawbacks: high volatility, often leading, non-optimal characteristic weighting, unwanted exposures to other factors, and high turnover due to abrupt inclusion/removal in the portfolio. FMPs are designed to be the most efficient (i.e., lowest risk) portfolios with unit exposure to a characteristic of interest.

---

**9.4 Trading in Idio Space**

In Section 4.3, we introduced the concepts of alpha spanned and alpha orthogonal. Alpha spanned are asset-expected returns attributable to co-movements with factor returns; alpha orthogonal are not explainable by factor returns. Because of Equation (4.2), Sharpe Ratio scales at least like $\sqrt{N}$. Because of this, alpha orthogonal is the golden currency in investing. How does one build a portfolio that exploits this alpha orthogonal, meaning that the optimal portfolio has zero factor exposures, is long alpha, is factor-neutral, and has minimum idiosyncratic risk, and they are encountered in practice. The first one is to build a portfolio that has an upper bound on volatility, maximizes expected returns, and has no factor exposures. By construction, the portfolio contains “pure alpha” and no factor-related PnL. The formulation is
(9.18)
$$ \max_{\mathbf{w}} \alpha^T \mathbf{w} $$
$$ \text{s.t. } \mathbf{B}^T \mathbf{w} = \mathbf{0} $$
$$ \mathbf{w}^T \Omega_\epsilon \mathbf{w} \le \sigma^2 $$
whose solution is[10]
$$ \tilde{\alpha}_\perp = (I_N - \mathbf{B}(\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{B}^T \Omega_\epsilon^{-1}) \alpha_\perp $$
$$ \mathbf{w}^* = \frac{\sigma}{\sqrt{\tilde{\alpha}_\perp^T \Omega_\epsilon^{-1} \tilde{\alpha}_\perp}} \Omega_\epsilon^{-1} \tilde{\alpha}_\perp $$
In Section 4.3.1, we built an FMP for a factor through two steps: first, orthogonalization; second, inverse-variance weighting. The portfolio construction for an alpha orthogonal portfolio is, indeed, identical: orthogonalization and inverse variance weighting. If all the asset idiosyncratic volatilities are identical, $\Omega_\epsilon$ is proportional to the identity, and the orthogonalization step is superfluous: $\tilde{\alpha}_\perp = \alpha_\perp$.

---

**9.5 Drivers of Information Ratio: Information Coefficient and Diversification**

What makes a good strategy before we are trading (i.e., ex ante), and we are living the dream, i.e., Equation (9.1)?
$$ \text{SR}^* = \sqrt{\alpha^T \Omega_r^{-1} \alpha} $$
A substantial part of this and of the next chapter is dedicated to the notion that we do not live in the dreamtime. Our forecasted returns and risk models are incorrect, and we should incorporate this knowledge in the investment process. A first step is to re-analyze the problem of maximizing the Sharpe Ratio. Start with the solution to the MVO problem, Equation (9.3):
$$ \mathbf{w}^* = \frac{\sigma}{\sqrt{\alpha^T \Omega_r^{-1} \alpha}} \Omega_r^{-1} \alpha $$
and assume that the covariance matrix $\Omega_r$ is accurate, admittedly a strong assumption. The expected realized return is
$$ E(\mathbf{r}^T \mathbf{w}^*) = \frac{\sigma}{\sqrt{\alpha^T \Omega_r^{-1} \alpha}} (\alpha^T \Omega_r^{-1} \alpha) = \sigma \sqrt{\alpha^T \Omega_r^{-1} \alpha} $$
$$ E(\mathbf{r}^T \mathbf{w}^*) = \frac{E(\mathbf{r}^T \mathbf{w}^*)}{\sqrt{\mathbf{w}^{*T} \Omega_r \mathbf{w}^*}} = \frac{\sigma \sqrt{\alpha^T \Omega_r^{-1} \alpha}}{\sigma} = \sqrt{\alpha^T \Omega_r^{-1} \alpha} $$
Recall the Information Coefficient, already introduced in Section 8.3.1:
$$ \text{IC} := \frac{E(\mathbf{r}^T \mathbf{w}^*)}{\sqrt{E(\mathbf{r}^T \mathbf{r})} \sqrt{E(\mathbf{w}^{*T} \Omega_r \mathbf{w}^*)}} = \frac{\alpha^T \Omega_r^{-1} \alpha}{\sqrt{\alpha^T \Omega_r^{-1} \Omega_r \Omega_r^{-1} \alpha} \sqrt{\alpha^T \Omega_r^{-1} \alpha}} $$
The important thing to know is that the Information Coefficient is a correlation. To see why, we need to transform variables:
(9.19)
$$ \tilde{\alpha} = \Omega_r^{-1/2} \alpha $$
So that the Information Coefficient can be rewritten in a more succinct form
$$ \text{IC}(\tilde{\alpha}, \mathbf{r}) = \frac{E(\tilde{\alpha}^T \Omega_r^{1/2} \mathbf{r})}{\sqrt{\tilde{\alpha}^T \tilde{\alpha}} \sqrt{E(\mathbf{r}^T \Omega_r \mathbf{r})}} $$
which can be interpreted as a cross-sectional uncentered correlation between z-scored alpha and z-scored returns.
We can simplify things further by proving that $E[\mathbf{r}^T \Omega_r^{-1} \mathbf{r}] = m$. The random vector $\mathbf{z} = \Omega_r^{-1/2} \mathbf{r}$ has the same covariance matrix as $\mathbf{r}$, where $\mathbf{z}$ is a standard multivariate normal.
$$ E(\mathbf{r}^T \Omega_r^{-1} \mathbf{r}) = E(\mathbf{z}^T \Omega_r^{-1/2} \Omega_r \Omega_r^{-1/2} \mathbf{z}) = E(\mathbf{z}^T \mathbf{z}) $$
$$ = \sum_{i=1}^m E(z_i^2) $$
$$ = m $$
Putting everything together, the $E(\mathbf{r}^T \mathbf{w}^*)$ is
$$ \text{SR} = \frac{\text{IC}(\tilde{\alpha}, \mathbf{r})}{\sqrt{m}} $$
This relationship goes back to Grinold (1989) and Grinold and Kahn (2000), who named it The Fundamental Law of Active Management.[11] It is often invoked by practitioners. In practice, users of the formula do not whiten returns and alphas in advance, instead they replace $\Omega_r$ by the diagonal $\Omega_\epsilon$. So, IC is the cross-sectional correlation between alpha and idiosyncratic returns (not total returns). The returns in units of volatility are the standardized idio returns (so that they have unit variance).
The Fundamental Law has several important implications. The first, and most obvious one, is that performance is driven by two factors. The first one is a measure of skill: the Information Coefficient. It is a myth that the prediction strength of the IC is low. It has been shown to be as high as 0.2 for some strategies, and depending on its predictive power, we should do so? This may not be the case in real life, depending on the specifics of your strategy. Many investors also have a notion of “idea velocity”, expressed as the number of forecasts $T^*$ per year. A high-er idea velocity increases, in principle, the Information Ratio. It is really difficult, however, to increase effectively the frequency of forecasts $T^*$.

**Insight 9.5: Information Coefficient and prediction regression**
Being a correlation, the IC is also naturally related to the predictive strength of our alphas, as measured by a cross-sectional regression. An important step in exploring alpha is to regress realized returns on the forecasted expected returns. We want to explain a fraction of residual returns using alpha. We form an efficient $\mathbf{z}$ that solves the following minimization problem:
$$ \min_{\mathbf{z}} \sum_{i,t} \frac{(r_{it} - \alpha_i z_t)^2}{\sigma_{\epsilon,it}^2} = \min_{\mathbf{z}} \| \tilde{\mathbf{r}} - \tilde{\alpha} \mathbf{z} \|_F^2 $$
The solution is given by $\mathbf{z}^* = \tilde{\mathbf{r}}^T \tilde{\alpha} / \|\tilde{\alpha}\|^2$ and the residual sum of squares is
$$ \|\tilde{\mathbf{r}}\|^2 - (\tilde{\mathbf{r}}^T \tilde{\alpha})^2 / \|\tilde{\alpha}\|^2 $$
, while the total sum of squares is $\|\tilde{\mathbf{r}}\|^2$. The coefficient of determination ($R^2$ squared) is, in expectation, equal to
$$ R^2 = \frac{(\tilde{\mathbf{r}}^T \tilde{\alpha})^2}{\|\tilde{\alpha}\|^2 \|\tilde{\mathbf{r}}\|^2} = (\text{IC})^2 $$
And we can link the coefficient of determination in predictive regressions to the Information Ratio:
$$ \text{IR} = \sqrt{R^2 n} $$
If there are $T^*$ investment periods in a year, the annualized Information Ratio has a convenient form as a function of per-period cross-sectional R-squared:
$$ \text{IR} = \sqrt{R^2 n T^*} = \text{IC} \sqrt{n T^*} $$
Otherwise stated, the annualized Information Ratio is equal to the Information Coefficient times the independent number of assets/forecasts in a year.[12]
[12] For the relationship between coefficient of determination and IR, see Chincarini and Kim (2006, 2022).
The Fundamental Law also connects IR to an ex-post measure (IC) that is interpretable as a correlation, which can be related to a spatial kind of regression (as per Insight 9.5).

---

**9.6 Aggregation: Signals versus Portfolios**

So far, we have considered the optimization of a single portfolio. That is, the entire family found by a quantitative portfolio manager. But life is far from being this easy. Let us consider two examples. In the first one, you have a fundamental portfolio manager managing a large team. In order to scale her book, the PM delegates to the analyst trading decisions for the stocks they cover. The portfolio of the entire team is then simply the aggregation of the individual portfolios. Is this a fully decentralized model, or is it a centralized model? What are the advantages? The analysts do their own research, summarize it into a thesis for each stock, and communicate to the PM, who converts them into positions. This is the centralized solution to the problem. Now, we produce the second example by going up one level: the large team is one of many teams belonging to a large hedge fund. The CIO of the hedge fund would like to deploy capital and would like to deploy one option available to the hedge fund is to take the portfolio positions of the individual teams, and increase them by a given percentage, say 50%. This can be directly performed at the level of an order’s submittal. This is possible for the decentralized solution. The first one is, however, not necessarily a bad choice. The second one is that the portfolio manager is a prediction machine. The alternative is the centralized solution: gather the signals from the teams and construct a portfolio. Figure 9.1 visualizes the two approaches.

[Image: Figure 9.1: Left: the decentralized solution to portfolio combination. Right: the centralized solution.]

A lot of ink rides on the decision to centralize versus decentralize portfolio construction! How should we organize large teams? How should we deploy new capital in the most efficient way? The rest of this section is devoted to giving a simplified answer to these questions. The conditions are very roughly, that the portfolio constructor is an MVO problem, and there are no transaction costs. Technical details matter. However, the rest of the section spells them out in detail and states the precise result. As for most of the results in this book, the theorem is interesting not because it hasn’t held in practice. It allows us to evaluate centralized solutions under which it holds, and the real-world situation, and identify areas where one solution may have the advantage.
Let us formalize the problem. There is an investment universe of $N$ stocks. We have $M$ portfolio managers working under a principal hedge fund manager. Without loss of generality each one has the same investment coverage of $N_m$ stocks. Every portfolio manager sets the same asset covariance matrix $\Omega_r$, and has a forecast $\alpha_m$.
We model the portfolio managers by assuming they are mean-variance maximizers. Without loss of generality PM produces a unit-volatility portfolio.
$$ \mathbf{w}_m = \frac{1}{\sqrt{\alpha_m^T \Omega_r^{-1} \alpha_m}} \Omega_r^{-1} \alpha_m $$
$$ \sigma_m := \sqrt{\alpha_m^T \Omega_r^{-1} \alpha_m} $$
Correspondingly, we have two solutions.
1.  Decentralized. PMs trade; the hedge fund manager estimates their Sharpe Ratios and return correlations, and combines their portfolios so as to maximize the overall Sharpe Ratio.
2.  Centralized. The hedge fund manager receives the signals, simulates their properties (e.g., Sharpe Ratio or IC), forms a combined signal, and trades a single Sharpe-maximizing portfolio.
We consider the two solutions.
1.  Decentralized solution. The hedge fund manager observes Sharpe Ratio $S_m$ for $m=1, \dots, M$ for each PM, as well as the correlation matrix $C$ for the $M$ portfolios. Consider $s_m = \text{diag}(S_1, \dots, S_M)$. The optimal dollar volatility allocated to strategy $m$ is $v_m = (C^{-1} \mathbf{s})_m$. The $M$ portfolios have unit volatility, hence the aggregated portfolio is $\mathbf{w}_{dec} = \sum_m v_m \mathbf{w}_m$.
2.  Centralized solution. The hedge fund manager receives signals and aggregates them as $\alpha_{agg} = \sum_m v_m \alpha_m$. We need to identify weights $v_m$. Let $\alpha_{agg} = E(\mathbf{r})$. The expected PnL of the portfolio is $\mathbf{A}^T \mathbf{w} = \sum_m v_m \alpha_m^T \mathbf{w}_m = \sum_m v_m \sigma_m^2$.
    The covariance of the portfolio is
    $$ (\alpha_m \sigma_m^2, \alpha_n \sigma_n^2) \Omega_r (\alpha_m \sigma_m^2, \alpha_n \sigma_n^2)^T = \sum_{m,n} v_m v_n \sigma_m^2 \sigma_n^2 C_{mn} $$
    We find the weights so that the hedge fund manager maximizes the Sharpe Ratio. Define $S_m = \sigma_m$. The maximization problem is
    $$ \max_{\mathbf{s}} \frac{\mathbf{s}^T \mathbf{x}}{\sqrt{\mathbf{x}^T C \mathbf{x}}} $$
    $$ \text{s.t. } \mathbf{x} \in \mathbb{R}^M $$
    The solution is
    $$ \mathbf{x}^* = C^{-1} \mathbf{s} \sqrt{\mathbf{s}^T C^{-1} \mathbf{s}} $$
    , so that
    $$ \mathbf{w}_{cen} = \frac{1}{\sqrt{\mathbf{s}^T C^{-1} \mathbf{s}}} \sum_m (C^{-1} \mathbf{s})_m \frac{1}{\sigma_m} \alpha_m $$
    The two portfolios $\mathbf{w}_{dec}$ and $\mathbf{w}_{cen}$ are identical, save for a multiplicative constant, which is not essential.
    This equivalency result is, in a sense, positive, because it suggests that, at least to a first approximation, we can decentralize portfolio construction decisions. What could go wrong? A lot. It is a good point to re-examine the assumptions:
    1.  First, we have ignored execution costs. It is possible that the centralized solution has the advantage, since it would net out opposite side positions of individual portfolio managers. Optimal execution turns the problem into a multi-period one, so that the analysis in this section does not apply.
    2.  Second, we have assumed that the portfolio managers solve an MVO problem; the formulation accommodates total return and idio return problems. In real-world implementations, there are side constraints that may differ by portfolio managers. Coordination problems may also differ among managers, and between them and the hedge fund manager.
    3.  When the individual units are not systematic, but rather discretionary portfolios managers, the portfolios produced by them are not generally MVO.
    4.  In many cases, the volatility allocation to the signals of the PMs in the centralized solution is not done using MVO. Heuristics, precommitments to individual PMs, and other constraints may play a role.
    Finally, especially in the latter case of discretionary managers, the alpha signal may not be communicated in a timely manner.
    In summary, the result suggests that in a real-world setting, provided that transaction costs do not dominate and that agents are MVO optimizers, the centralized solution should not dramatically dominate the decentralized one. The emphasis is on “suggests”; more research is (always) needed.

---

**9.7 Appendix**

**9.7.1 Some Useful Results From Linear Algebra**

Spiked covariance matrices are the sum of a full-rank (possibly diagonal) matrix and a low-rank matrix. For this class of matrices, there are useful, computationally efficient formulas that relate the inverse and the determinant of the matrix to those of the constituents. These are the Woodbury-Sherman-Morrison Lemma and the Matrix Determinant Lemma.

**Woodbury-Sherman-Morrison Lemma.**
Useful to compute the inverse of a matrix (e.g., min-variance portfolio and log-likelihood)
(9.20)
$$ (D + B B^T)^{-1} = D^{-1} - D^{-1} B (I + B^T D^{-1} B)^{-1} B^T D^{-1} $$

**Determinant Lemma.**
Useful in log likelihood calculations.
(9.21)
$$ \det(D + B B^T) = \det(D) \det(I + B^T D^{-1} B) $$

**9.7.2 Some Portfolio Optimization Problems**

**Example 9.1 (Maximize expected return subject to a vol constraint and linear homogeneous equalities):**
$$ \max_{\mathbf{w}} \alpha^T \mathbf{w} $$
$$ \text{s.t. } \mathbf{B}^T \mathbf{w} = \mathbf{b} $$
$$ \mathbf{w}^T \Omega_r \mathbf{w} \le \sigma^2 $$
The solution $\mathbf{w}^*$ to this problem is given by
$$ \Pi := I_N - \mathbf{B}(\mathbf{B}^T \Omega_r^{-1} \mathbf{B})^{-1} \mathbf{B}^T \Omega_r^{-1} $$
$$ \tilde{\alpha} := \Pi \alpha $$
$$ \mathbf{w}^* = \frac{\sigma}{\sqrt{\tilde{\alpha}^T \Omega_r^{-1} \tilde{\alpha}}} \Omega_r^{-1} \tilde{\alpha} $$

**Example 9.2 (Minimum-variance portfolio subject to linear equalities):**
(9.22) $\min_{\mathbf{w}} \mathbf{w}^T \Omega_r \mathbf{w}$
(9.23) s.t. $\mathbf{B}^T \mathbf{w} = \mathbf{b}$
The solution is
$$ \mathbf{w}^* = \Omega_r^{-1} \mathbf{B} (\mathbf{B}^T \Omega_r^{-1} \mathbf{B})^{-1} \mathbf{b} $$
Of special interest is the case where
$$ \Omega_r = B \Omega_f B^T + \Omega_\epsilon $$
. The first term is constant, so the objective is $\mathbf{w}^T \Omega_\epsilon \mathbf{w}$, and the
$$ \mathbf{w}^* = \Omega_\epsilon^{-1} \mathbf{B} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{b} = \mathbf{P} \mathbf{b} $$
where $\mathbf{P}$ is the matrix whose column vectors are the FMPs associated to the factor model. These are introduced in Section 9.3.1.

**9.7.3 Optimality of FMPs**

We now prove that FMPs emerge naturally from certain assumptions about returns and FMP composition, which we use in Section 9.3.1.

**Theorem 9.1:**
Consider a sequential model $S(\mathbf{B}, \Omega_f, \Omega_\epsilon)$. Assume that
1.  Alpha orthogonal is zero, so that $\alpha = \mathbf{B}\lambda$, for some $\lambda \in \mathbb{R}^m$.
2.  Idiosyncratic variance converges to zero in norm:
    $$ \lim_{N \to \infty} \| (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \| = 0 $$
Then in the limit, the Sharpe-optimizing portfolio is a weighted sum of the FMPs, and the weights themselves solve an MVO in factor space.
$$ \mathbf{w}^* = \mathbf{P} \mathbf{u}^* $$
where: $\mathbf{u}^* = \underset{\mathbf{u} \in \mathbb{R}^m}{\arg \max} \lambda^T \mathbf{u} - \frac{1}{2\gamma} (\mathbf{u}^T \Omega_f \mathbf{u})$, $U = (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1/2} \mathbf{B}^T \Omega_\epsilon^{-1}$
for some $\gamma > 0$. The Sharpe Ratio of the optimal portfolio is equal to
$$ \text{SR}^* = \sqrt{\lambda^T \Omega_f^{-1} \lambda} $$
The second condition deserves further scrutiny. A simple interpretation is easy to check. For the FMPs, the covariance matrix of the idiosyncratic PnL is $(\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1}$. The condition states that the idiosyncratic variance of the FMPs goes to zero.
Proof:
Start with the Sharpe-optimizing portfolio problem, Equation (9.3):
$$ \mathbf{w}^* = \frac{\sigma}{\sqrt{\alpha^T \Omega_r^{-1} \alpha}} \Omega_r^{-1} \alpha $$
$$ \Omega_r^{-1} = (\mathbf{B} \Omega_f \mathbf{B}^T + \Omega_\epsilon)^{-1} = \Omega_\epsilon^{-1} - \Omega_\epsilon^{-1} \mathbf{B} (\Omega_f^{-1} + \mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{B}^T \Omega_\epsilon^{-1} $$
The second identity is the Woodbury-Sherman-Morrison Lemma.
Now we perform a first-order expansion. Notice that:[13]
$$ (\Omega_f^{-1} + \mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} = (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} - (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \Omega_f^{-1} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} + O(\|\Omega_f^{-1}\|^2) $$
So we perform a first-order approximation of the inverse:[14]
$$ \Omega_r^{-1} \approx \Omega_\epsilon^{-1} - \Omega_\epsilon^{-1} \mathbf{B} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{B}^T \Omega_\epsilon^{-1} + \Omega_\epsilon^{-1} \mathbf{B} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \Omega_f^{-1} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{B}^T \Omega_\epsilon^{-1} $$
Replace the expression in the solution $\mathbf{w}^*$:
$$ \mathbf{w}^* = \mathbf{P} (\Omega_f^{-1} + \mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{B}^T \Omega_\epsilon^{-1} \alpha $$
$$ \approx \mathbf{P} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{B}^T \Omega_\epsilon^{-1} \alpha - \mathbf{P} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \Omega_f^{-1} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} \mathbf{B}^T \Omega_\epsilon^{-1} \alpha $$
$$ = \mathbf{P} \mathbf{u}^* $$
The second PnL of the optimal solution is $\mathbf{u}^{*T} \Omega_f \mathbf{u}^*$. The factor exposure of the optimal solution is $\mathbf{u}^*$. The factor variance is $\mathbf{u}^{*T} \Omega_f \mathbf{u}^*$. The idiosyncratic variance is
$$ (\mathbf{w}^*)^T \Omega_\epsilon \mathbf{w}^* = \mathbf{P}^* \Omega_f \mathbf{P} $$
$$ \approx (\mathbf{u}^*)^T (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} (\mathbf{u}^*) $$
and is zero in the limit, per the second assumption. The Sharpe Ratio is $\sqrt{\lambda^T \Omega_f^{-1} \lambda}$.

**9.7.4 Single Factor Covariance Matrix Updating**

Here we prove a basic result on the updated factor covariance matrix when adding a new factor. The analysis assumes that the parameters $\mathbf{B}_m, \Omega_f, \Omega_\epsilon$ of the factor model are constant, and so is the vector $\mathbf{a}$ or characteristics that we are using to augment the factor model. As in Section 9.3.2.

**Theorem 9.2:**
Let the loadings, factor covariance matrix, and idiosyncratic of a factor model be $\mathbf{B}_m, \Omega_{f,m}$ and $\Omega_{\epsilon,m}$. Let $\mathbf{a} \in \mathbb{R}^N$ be a vector of characteristics. Define
$$ \mathbf{b}_{m+1} = (I_N - \mathbf{B}_m (\mathbf{B}_m^T \Omega_{\epsilon,m}^{-1} \mathbf{B}_m)^{-1} \mathbf{B}_m^T \Omega_{\epsilon,m}^{-1}) \mathbf{a} $$
The factor covariance matrix associated to the model with loadings $[\mathbf{B}_m, \mathbf{b}_{m+1}]$ is given by
$$ \hat{\Omega}_f \approx \begin{bmatrix} \hat{\Omega}_{f,m} & \mathbf{0} \\ \mathbf{0}^T & \sigma_{m+1}^2 \end{bmatrix} $$
Proof:
Let the factor return of the new factor be
$$ \hat{f}_{m+1,t} = \frac{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{r}_t}{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}} $$
$$ = \mathbf{v}_{m+1,t}^T \mathbf{r}_t $$
as in Equation (9.14). We assume that the new factor does not have a big impact on the idiosyncratic returns in the sense that
$$ \text{cov}(\hat{f}_{m+1,t}, \hat{f}_{k,t}) \approx 0, \quad k \le m $$
, so that $\Omega_f$ is approximately unchanged by the addition of the new factor. Then the volatility of the new factor is
$$ \hat{\sigma}_{m+1}^2 = (\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1})^{-1} $$
$$ E[\hat{f}_{m+1,t}^2] = \frac{E[(\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{r}_t)^2]}{(\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1})^2} $$
$$ \hat{\sigma}_{m+1}^2 = \frac{\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \Omega_r \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1}}{(\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1})^2} $$
$$ = (\mathbf{b}_{m+1}^T \Omega_{\epsilon,m}^{-1} \mathbf{b}_{m+1})^{-1} $$
We show that $f_{m+1,t}$ is approximately uncorrelated to the first $m$ factors. The column vector of the first $m$ ex-factor returns is $\mathbf{P}^T \mathbf{r}_t$.
$$ E[\hat{f}_{m+1,t} \mathbf{P}^T \mathbf{r}_t] = E[\mathbf{v}_{m+1,t}^T \mathbf{r}_t \mathbf{r}_t^T \mathbf{P}] $$
$$ = \mathbf{v}_{m+1,t}^T (\mathbf{B} \Omega_f \mathbf{B}^T + \Omega_\epsilon) \mathbf{P} $$
$$ = \mathbf{v}_{m+1,t}^T \mathbf{B} \Omega_f (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} + \mathbf{v}_{m+1,t}^T \Omega_\epsilon \Omega_\epsilon^{-1} \mathbf{B} (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})^{-1} $$
$$ = \mathbf{P}^T \Omega_f \mathbf{P} \mathbf{v}_{m+1,t} $$
The last equality follows from the orthogonality of $\mathbf{b}_{m+1}$. The correlation between FMPs is given by the idiosyncratic component of their returns, which should be small if the factors are diversified.

**The Takeaways**

1.  Mean-Variance Optimization (MVO) is a foundational approach to portfolio construction due to its interpretability, data efficiency, and computational simplicity in single-period settings.
2.  Factor models enable systematic decomposition of portfolio risk and return by representing asset covariance through factors, simplifying the optimization process and enhancing interpretability.
3.  Factor-Mimicking Portfolios (FMPs) are valuable instruments for factor-based investing, offering efficient trading strategies focused on specific characteristics.
4.  The Information Ratio (IR) is influenced by the quality of signals (Information Coefficient) and the degree of diversification, providing a comprehensive measure of performance potential.
5.  Portfolio construction can be approached via a centralized or decentralized approach when the MVO principles, but practical differences like transaction costs and constraints may make one approach preferable.

**Notes**

1.  [1] Extensive treatments of convex optimization are Boyd and Vandenberghe (2004), Bazaraa et al. (2006), and Luenberger and Ye (2008); and finance-oriented optimization textbooks are Cornuejols et al. (2006), Palomar (2014).
2.  [2] On justification of the mean-variance approach to portfolio optimization, see Huang and Litzenberger (1988). Both cover the standard cases of exponential and quadratic utilities. A number of textbooks exist covering portfolio construction. A classic is Grinold and Kahn (1999); see also Qian et al. (2007), Isichenko (2018). On the definition of the Sharpe Ratio see Lo (2002), and for a comprehensive and detailed treatment of the (Sharpe) ratio and its uses, there is
3.  [3] We ignore the term $\alpha_k$, both out of simplicity and because it is very small.
4.  [4] The subject will be covered in Section 10.2.1.
5.  [5] See Section 4.7.2.
6.  [6] See Sections 4.7.2 and 4.7.3.
7.  [7] See Sections 4.7.2 and 4.7.3.
8.  [8] Devise it as an exercise, or see Section 4.7.3.
9.  [9] Prove this step-by-step in Exercise 4.3.
10. [10] Quite an important name for a “law”. And why not? Nobody had thought of using this cute acronym before. I can imagine Grinold playing air guitar and screaming to Iron Maiden when he was writing this mighty paper.
11. [11] For example, Modigliani-Miller’s theorem and Merton’s result on the equivalence of an option and a replicating portfolio.
12. [12] Given square matrices $A, B$, the inequality on spectral norms $|AB| \le |A| |B|$ holds.
13. [13] The first-order expansion (from von Neumann’s series) is $(I+A)^{-1} \approx I-A$.
   

Okay, here is the Markdown compilation for Chapter 10.

```markdown
Everand

