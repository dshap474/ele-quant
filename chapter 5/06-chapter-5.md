**Chapter 5**
**Evaluating Risk**

**The Questions**

1.  Given the absence of a single performance measure for benchmarking factor models, how should one proceed and select the best model for a specific use-case?
2.  What are the robust loss functions suitable for evaluating volatility predictions in factor models?
3.  What are the tests suitable for testing the performance of covariance matrices for portfolio optimization?
4.  What are the advantages and limitations of different approaches?
5.  What additional tests should we perform to ensure that models can be used in production environments?

There are dozens of papers and documents extolling the virtues of commercial models. Many alternative models. Several asset classes. All combinations of geographies. And, of course, many vendors. However, there are hardly any papers laying out theoretically motivated procedures that test risk properties of factor models that allow practitioners to choose among them. This is a void that I believe is worth plumbing. In the words of E. E. Cummings, “Listening is Love the Earth; nobody wants to help Mom do the dishes.” This is a chapter about doing dishes. It relies on two simple principles. First, the metrics that we use should be related as much as possible to real applications. We care about accuracy of volatility forecasts, and realized volatility of optimized portfolios. We also want to measure these quantities in a realistic setting. The second principle follows from the observation that there is not a single performance measure on which we benchmark factor models. It follows that it is possible that a single “best” model may not exist, because it could be that a single model outperforms all the others on all metrics. This should not be overly surprising. Our goal is not to find the best, but to concentrate our efforts on our use-case, or at least prioritize for it, and find the best model for the task.

The remainder of the chapter is organized around three families of metrics: those aimed at evaluating the covariance matrix, those aimed at evaluating the precision matrix, and those aimed at evaluating the model suitability for secondary tasks.

---

**5.1 Evaluating the Covariance Matrix**

**5.1.1 Robust Loss Functions for Volatility Estimation**

A major application of a factor model is volatility estimation. The quality of volatility predictions is one that has been at the core of research in the development of risk models. It is often framed as a problem of point prediction. At a certain time $t$ (i.e., the volatility prediction made available at time $t-1$ for time interval $[t-1, t]$), there is a measure of the quality of the volatility predictions is given by a loss function
$$ L_t = \frac{1}{N} \sum_{i=1}^N L(\hat{\sigma}_{it}^2, \sigma_{it}^2) $$
where $\hat{\sigma}_{it}$ is an empirical estimate of the observed volatility $\sigma_{it}$, the realized volatility returns of asset $i$ over period $t$. This loss function $L(x,y)$ is non-negative, and equal to zero if and only if $x=y$. In the metrics station above, we use a volatility proxy $r_{it}^2$ instead of the unknown true volatility $P_t$. Hansen and Lunde (2006a) introduce a concept of rank robustness for losses: if we have two alternative volatility forecasts $\hat{\sigma}_{it}^{(1)}$ and $\hat{\sigma}_{it}^{(2)}$, $\hat{\sigma}_{it}^{(1)}$ is better than the other using an unbiased volatility proxy if and only if one is better than the other using the true volatility. That is,
$$ L(\hat{\sigma}_{it}^{(1)}, \sigma_{it}^2) \le L(\hat{\sigma}_{it}^{(2)}, \sigma_{it}^2) \iff L(\hat{\sigma}_{it}^{(1)}, \hat{r}_{it}^2) \le L(\hat{\sigma}_{it}^{(2)}, \hat{r}_{it}^2) $$
Patton and Sheppard (2009); Patton (2011) completely characterize these loss functions and show that these two are robust:
$$ \text{QLIKE}(\sigma, \tau) := \frac{1}{T} \sum_{t=1}^T \left( \frac{r_t^2}{\hat{\sigma}_t^2} - \log\left(\frac{r_t^2}{\hat{\sigma}_t^2}\right) - 1 \right) $$
$$ \text{MSE}(\sigma, \tau) := \frac{1}{T} \sum_{t=1}^T \left( \frac{r_t^2}{\hat{\sigma}_t^2} - 1 \right)^2 $$
QLIKE is (save for constants) the negative of the log-likelihood of the normal distribution. These two loss functions are increasingly being used in place of the Bias statistic.[1] Their graphs are shown in Figure 5.1.

[Image: Figure 5.1: QLIKE and MSE comparison. Notice that QLIKE is skewed, with higher losses when the realized variance is smaller than the estimated variance. Plot shows QLIKE (red) and MSE (blue) loss functions vs r_t^2 / sigma_hat_t^2.]

**5.1.2 Application to Multivariate Returns**

The loss functions QLIKE and MSE apply to univariate returns, not to covariance matrices. Below are a few ways to adapt the univariate setting to a multivariate one.

*   **Production Strategies.**
    If strategies are already running, a straightforward and necessary test is to evaluate their simulated performance under different factor models. QLIKE and MSE are important and should be checked jointly with metrics that are important for the portfolio manager, like Sharpe Ratio or PnL. It is important that the covariance matrix is tailored to the production strategy. Ideally, a portfolio should be generated on the basis of the factor model itself. If a portfolio is generated using factor model A and then tested on model B, the test will be marred by this asymmetry.
*   **Average-Case Analysis.**
    An alternative approach is to estimate the expected loss, where the expectation is taken over a distribution of portfolios as well as of asset returns. For the distribution of returns, we use the empirical measure $P$ of historical returns; for the distribution of portfolios, we may choose a simple one, like uniform on a sphere (see Procedure 5.1). Then we estimate
    $$ E_{P,W} [L(W' \hat{\Omega} W, W' \Omega W)] $$
    There are a few drawbacks to this approach. First, there is a degree of arbitrariness in choosing a portfolio distribution. The actual distribution of portfolios is almost certainly not uniform; and it is not even warranted that the distribution of alphas is uniform. Second, it is computationally expensive. We are averaging in high dimensional spaces, with $W$ varying in size from $10^2$ to $10^4$ in typical models. Consequently, issues such as calculation errors and convergence criteria, become important. An approximation is to select a portfolio basis, say, $N$ portfolios $W_1, \dots, W_N$, and then apply an “average” case analysis to these $N$ portfolios. A special case of this approach is that of eigenportfolios (see Section 5.2.1), i.e., $U S U'$ where $U$ are the eigenportfolios (equal to the columns of $U$). This is more computationally tractable. One important drawback is that this average loss is not independent of the choice of the portfolio; in fact, it is quite sensitive to it. Even if we restrict our choice to an orthonormal basis, the performance of the measured portfolio still depends on the basis. Since the choice of an appropriate basis cannot be easily justified based on principles, the outcome is arbitrary.

    **Procedure 5.1: Random portfolios average variance testing**
    1.  Inputs: candidate covariance matrices $\hat{\Omega}_k$, and returns $r_t$ for $t=1, \dots, T$.
    2.  Set $L_{tot} = 0$.
    3.  Set $W \sim N(0, I_N)$, $W \leftarrow W/\|W\|$.
    4.  Choose $k$ uniformly at random in $1, \dots, K$.
    5.  $L_{tot} \leftarrow L_{tot} + L(r_t' W, W' \hat{\Omega}_k W, W' \Omega W)$
    6.  Set $N_{sim} \leftarrow N_{sim} + 1$.
    7.  If $L(r_t' W, W' \hat{\Omega}_k W, W' \Omega W) / L_{tot} \ge \epsilon_{tol}$
        go to Step 3.
    8.  Output: $L_{avg} = L_{tot} / N_{sim}$.
*   **Worst-Case Under/Over Prediction.**
    Yet another approach is to estimate the worst-case loss function:
    $$ \max_W E_P [L(r'W, W' \hat{\Omega} W, W' \Omega W)] $$
    $$ \text{s.t. } \|W\| \le 1 $$
    The problem with this approach is that the objective function (be it QLIKE or MSE) is not convex. When the number of assets is large, the problem is not computationally tractable.

    **Procedure 5.2: Worst-case variance testing**
    1.  Inputs: candidate covariance matrices $\hat{\Omega}_k$, and returns $r_t$ for $t=1, \dots, T$. Loss function $L$.
    2.  Set $L_{max} = 0, N_{iter} = 0$.
    3.  Set $W \sim N(0, I_N)$, $W \leftarrow W/\|W\|$.
    4.  Choose $k$ uniformly at random in $1, \dots, K$.
    5.  Set
        $$ W \leftarrow W - \eta_{iter} \nabla_W L(r_t' W, W' \hat{\Omega}_k^T W, W' \Omega W) $$
    6.  Set $N_{iter} \leftarrow N_{iter} + 1$.
    7.  If $L(r_t' W, W' \hat{\Omega}_k^T W, W' \Omega W) / L_{tot} \ge \epsilon_{tol}$
        go to Step 3.
    8.  Output: $L_{max} = \max_k L(r_t' W, W' \hat{\Omega}_k^T W, W' \Omega W)$.

    None of the above dominates the others. Whenever available, production strategies are always tested against alternative approaches. Average- and worst-case analyses are both computationally very demanding. Moreover, in the case of average-case analysis the result depends on the assumption on portfolio distribution.
*   **Leading Alpha, MVO Portfolios.**
    Another option is to construct portfolios based on the actual leading alphas of the securities. This scheme has the advantage to use the predictiveness of the strategy for “relevant” portfolios and is described in Procedure 5.3. After all, volatility prediction matters only if we have alpha in the first place. If we don’t, then we have other problems to be worried about.

    **Procedure 5.3: Realized alpha variance testing**
    1.  Inputs: candidate covariance matrices $\hat{\Omega}_k$, and returns $r_t$ for $t=1, \dots, T$. Loss function $L$.
    2.  Set $L_{tot} = 0$.
    3.  For each $t=0, \dots, T-\tau$, let
        $$ \alpha_t := \frac{1}{\tau} \sum_{s=t+1}^{t+\tau} r_s $$
        $$ W_t := \hat{\Omega}_{k,t}^{-1} \alpha_t $$
    4.  $L_{tot} := L_{tot} + L(r_{t+\tau+1}' W_t, W_t' \hat{\Omega}_{k, t+\tau+1}^T W_t)$
    5.  Output:
        $$ L := L_{tot} / (T-\tau+1) $$
    An advantage of this approach is that it can easily be augmented. For example, we could test the performance on portfolios with added noise:
    $$ W := \hat{\Omega}_{k,t}^{-1} (\alpha_t + \eta_t), \quad \eta_t \sim N(0, \sigma_\eta^2 I_N) $$
*   **Distribution Likelihood.**
    An alternative that does not depend on the portfolio choice is to use the log-likelihood for the zero-mean multivariate normal distribution, applied to the returns of the estimation universe. Modulo constant terms, the negative log-likelihood, $L$, is proportional to:
    (5.2)
    $$ \text{QDIST} = \sum_t (\mathbf{r}_t' \hat{\Omega}_{k,t}^{-1} \mathbf{r}_t + \log |\hat{\Omega}_{k,t}|) + N \log(2\pi) $$
    **Exercise 5.1**
    In their survey paper, Patton and Sheppard (2009) propose a multivariate test for $\Omega_k$ using a proxy estimate for the true covariance matrix $\hat{\mathbf{H}}_t$. The loss function they propose is
    $$ L = \sum_t \text{trace} \left( (\hat{\Omega}_{k,t}^{-1} - \hat{\mathbf{H}}_t^{-1}) \hat{\mathbf{H}}_t (\hat{\Omega}_{k,t}^{-1} - \hat{\mathbf{H}}_t^{-1}) \hat{\mathbf{H}}_t \right) - T \log |\hat{\Omega}_{k,t}^{-1} \hat{\mathbf{H}}_t| $$
    Show that, when $\hat{\mathbf{H}}_t = r_t r_t'$, $L$ is equivalent to QDIST, i.e.,
    $L = a \times \text{QDIST} + b$
    for some positive constant $a$.

---

**5.2 Evaluating the Precision Matrix**

**5.2.1 Minimum Variance Portfolios**

As we have seen repeatedly throughout the book, the quality of a factor model is related to the accuracy of its precision matrix. We propose two tests. The first one is based on a well-known portfolio construction method. We consider a very simple example: construct a portfolio $\mathbf{w}$ of minimum variance and with unit net market value $\sum_i w_i = 1$. This is the ex-ante minimum variance portfolio; the realized variance will differ. The intuition is that a “better” covariance matrix will result in a realized variance. We make this intuition rigorous, and generalize to the case where the portfolio has a given expected return or arbitrary factor risk. Let $\hat{\Omega}_k \in \mathbb{R}^{N \times N}$ be a candidate covariance matrix and $\Omega_t$ be the true covariance matrix. Let $\hat{\mathbf{w}}_k = \hat{\Omega}_k^{-1} \mathbf{1}_N$, and solve the risk-minimization problem
$$ \min_{\mathbf{w}} \mathbf{w}' \hat{\Omega}_k \mathbf{w} $$
$$ \text{s.t. } \mathbf{1}_N' \mathbf{w} = 1 $$
and let $\mathbf{w}_k(\hat{\Omega}_k)$ be its solution. We compare the realized variance of the portfolio $\text{var}(\mathbf{w}_k(\hat{\Omega}_k), \Omega_t)$. Then, the realized volatility of portfolio $\mathbf{w}_k(\hat{\Omega}_k)$ is greater than that of $\mathbf{w}_k(\Omega_t)$, and the two are identical if and only if $\hat{\Omega}_k = \Omega_t$. This is Theorem 5.1 in the Appendix (Section 5.4). A way to apply this result is as follows: fix $\hat{\Omega}_k = \hat{\Omega}_A$, and define $\mathbf{w}_A = \mathbf{w}_k(\hat{\Omega}_A)$. Then compare $\text{var}(\mathbf{w}_A, \Omega_t)$ with $\text{var}(\mathbf{w}_B, \Omega_t)$. This comparison of realized variance of minimum variance portfolios is by far the realized Sharpe Ratio for a certain strategy.

We can use all the portfolio-dependent schemes introduced for volatility tests to evaluate the precision matrix. The realized variance acts as a loss function.

**5.2.2 Mahalanobis Distance**

There is another test that is portfolio-independent and that involves the precision matrix only. The Mahalanobis distance is defined for a multivariate zero-mean random vector $\mathbf{r}$ and an associated covariance matrix $\Omega_r$ as
$$ d(\mathbf{r}, \Omega_r) = \sqrt{\mathbf{r}' \Omega_r^{-1} \mathbf{r}} $$
. For Gaussian returns and the true covariance matrix, $d_t^2$ is distributed according to a chi-squared distribution with $N$ degrees of freedom.[2] One test is then
$$ \text{MALV} := \frac{1}{T} \sum_{t=1}^T d(\mathbf{r}_t, \hat{\Omega}_{k,t})^2 = \frac{1}{T} \sum_{t=1}^T \mathbf{r}_t' \hat{\Omega}_{k,t}^{-1} \mathbf{r}_t $$
$$ \text{MALV} := \text{var} (P_1, \dots, P_T) $$
The lower the value of MALV($\hat{\Omega}_k$), the better the performance of the precision matrix. If the variance is very large (say, if $d_t^2 / N \gg 1$), the value of the test is not very useful for evaluating precision, preferring volatility. We don’t primarily care about the constant in this test, because a volatility test should address that issue. A different way to interpret the result is the following: if returns are Gaussian, then they are distributed as $N(0, \hat{\Omega}_k)$ with $\hat{\Omega}_k$ constant. The Mahalanobis distance is $N$ times the mean of $d_t^2$. Its mean is $N$ and standard deviation $\sqrt{2N}$. For large $N$, $d_t^2 \approx N$. If $\hat{\Omega}_k$ is constant (i.e., $\hat{\Omega}_{k,t} = \hat{\Omega}_k$), this means MALV as follows:
$$ \hat{\mu} = \frac{1}{T} \sum_{t=1}^T \mathbf{r}_t $$
$$ \hat{\Sigma} = \frac{1}{T} \sum_{t=1}^T (\mathbf{r}_t - \hat{\mu})(\mathbf{r}_t - \hat{\mu})' - \hat{\nu} $$
$$ \approx \frac{1}{T} \sum_{t=1}^T (\mathbf{r}_t' \hat{\Omega}_k^{-1} \mathbf{r}_t - N)^2 - \hat{\nu} $$
$$ \approx \frac{1}{T} \sum_{t=1}^T (\mathbf{r}_t' \hat{\Omega}_k^{-1} \mathbf{r}_t - N)^2 $$
The approximate equality results from the fact that we have replaced $\hat{\Omega}_k$ with its expected value. We see here an instance of
$$ \Omega_{k,t}^{-1/2} \mathbf{r}_t \sim N(0, I_N) - \hat{\mu}_k^T \hat{\Omega}_k^{-1} \hat{\mu}_k $$
its zero by multiplying the left and right by standard multivariate Gaussian random vectors. In Section 10.3, we will see that this matrix difference is also responsible for bounding the Sharpe Ratio efficiency.

---

**5.3 Ancillary Tests**

In addition to performance measures on the ex-ante covariance matrix $\hat{\Omega}_k$ and its inverse, we want to verify that the model performs well at tasks that are indirectly related to portfolio construction and hedging. We consider two specifically. The first one is model turnover. Changes over time in the data we have at hand affect the transaction costs of our strategies. There are three major drivers of such changes. The first is the time-varying nature of expected returns, i.e., alpha. However, the second and third drivers are factor exposures and factor volatilities. We need to verify that these changes are not too large. The second is due to changes in the market impact function itself. There are exceptional periods in which trading activity changes rapidly, and so does the cost of trading. However, in general, this is the most stable component in portfolio construction and we assume it is accurately known. The last driver of change is the factor return, which is the output of the following subsection.

**5.3.1 Model Turnover**

Turnover is an intrinsic property of a model. It is the property of a production trading strategy. However, it may make sense to have a given estimate of turnover. For example, in an ideal setting in which there are no transaction costs, we may want to target a constant factor exposure level, for factor $k$. In this setting, it is optimal to trade Factor-Mimicking Portfolios (FMPs). The FMP is the portfolio that provides the best approximation to the true factor returns of the model. They are defined in Section 7.1.1. The weights of these portfolios are the column vectors of the matrix
$$ \mathbf{P}_t := \Omega_{\epsilon,t}^{-1} \mathbf{B}_t (\mathbf{B}_t' \Omega_{\epsilon,t}^{-1} \mathbf{B}_t)^{-1} $$
$$ \mathbf{w}_t := \mathbf{P}_t \mathbf{b} $$
Portfolio turnover is driven by the changes in $\mathbf{P}_t$. We then define a simple measure of quadratic turnover as the time series average
$$ \text{turnover}_T := \frac{1}{T} \sum_{t=1}^T \| \mathbf{P}_t - \mathbf{P}_{t-1} \|_F^2 $$
where $\| \cdot \|_F$ is the Frobenius norm of a matrix. This definition has the advantage of being simple and intuitive. It is positive definite, at the cost of losing generality. If we have an indication of the target exposures $\mathbf{b}_t$ of our strategy, and we have a trading cost function $TC(\cdot, \cdot)$, then a more accurate measure of turnover would be
$$ \text{turnover}_{TC} := \frac{1}{T} \sum_{t=1}^T TC(\mathbf{P}_t - \mathbf{P}_{t-1}, \mathbf{b}_t) $$

**5.3.2 Testing Betas**

A practical application of risk models is to produce predicted betas of investor portfolios versus some benchmark. A simple instance is predicting the beta of a long-only portfolio of highly liquid stocks versus the S&P500. This beta is an FMP. An even less obvious example is that of beta to a thematic portfolio. For example, a bank has generated a thematic portfolio that describes an industrial or consumption trend that is expected to occur at that point in time. We want to measure the beta of this portfolio to this thematic portfolio. In all of these cases, we want to make sure that the predicted beta matches the ex-post beta, in the sense that it exhibits low discrepancy to the realized beta. Therefore, as part of the evaluation of the risk model, we want to include tests on betas. For simplicity, consider the case of a single-instrument portfolio (FMP, or thematic portfolio). The vector of predicted betas of each asset to this reference portfolio is
$$ \hat{\beta}_t(\mathbf{w}) = \frac{\mathbf{w}' \hat{\Omega}_{k,t} \mathbf{w}_{ref}}{\mathbf{w}_{ref}' \hat{\Omega}_{k,t} \mathbf{w}_{ref}} $$
The exponentially weighted empirical mean $\hat{\Omega}_{k,t}$, and realized beta vector is given by
$$ \hat{\Omega}_{k,t} = \frac{1 - e^{-T/\tau}}{1 - e^{-1/\tau}} \sum_{s=0}^{T-1} e^{-s/\tau} \mathbf{r}_{t-s} \mathbf{r}_{t-s}' $$
$$ \hat{\beta}_t(\mathbf{w}) := \frac{\mathbf{w}' \hat{\Omega}_{k,t} \mathbf{w}}{\mathbf{w}' \hat{\Omega}_{k,t} \mathbf{w}} $$
We measure the beta accuracy as
$$ \text{BETAERR}(\mathbf{w}) = \sum_t \| \hat{\beta}_t(\mathbf{w}) - \beta_t \|^2 $$
and employ BETAERR as an ancillary measure of accuracy.

---

**5.3.3 Coefficient of Determination?**

A very popular way to summarize the performance of a factor model is to measure the average coefficient of determination (or $R^2$) of the weighted cross-sectional regression. It is defined as 1 minus the ratio of the sum-of-squared residual sum of squares and the total sum of squares:
$$ R^2 = 1 - \frac{\sum_{t=1}^T \mathbf{e}_t' \Omega_{\epsilon,t}^{-1/2} \Omega_{\epsilon,t}^{-1/2} \mathbf{e}_t}{\sum_{t=1}^T (\mathbf{r}_t - \bar{\mathbf{r}}_t)' \Omega_{\epsilon,t}^{-1/2} \Omega_{\epsilon,t}^{-1/2} (\mathbf{r}_t - \bar{\mathbf{r}}_t)} = 1 - \frac{\sum_{t=1}^T \mathbf{e}_t' \Omega_{\epsilon,t}^{-1} \mathbf{e}_t}{\sum_{t=1}^T (\mathbf{r}_t - \bar{\mathbf{r}}_t)' \Omega_{\epsilon,t}^{-1} (\mathbf{r}_t - \bar{\mathbf{r}}_t)} $$
Since the idiosyncratic covariance matrix $\Omega_{\epsilon,t}$ is not known in advance, a proxy is used in its place, as described in Chapter 4 and 7. The estimated factor returns $\hat{\mathbf{f}}_t$ are the coefficients of the cross-sectional regression in period $t$. A high coefficient of determination is interpreted as a positive attribute of the model since, in the case of factor models, most of the variability of returns is due to common factors. First, there is a “look-ahead” estimate of $\Omega_{\epsilon,t}$. We cannot possibly estimate the performance of the model on a holdout sample because the estimated coefficients, the factor returns, must be estimated in every period. This makes the statistic sensitive to data snooping. Second, even if they keep constant over time, the quality of the model (the number of factors) and the choice of the successive manipulations and adjustments to the loading matrices $\mathbf{B}_t$. The naive $R^2$ is always improved by adding factors (even with random loadings) to an existing model, although the fact that more refined metrics, like AIC and BIC, provide more tools to use. It is not possible to answer this question in general. Another solution to the use of $R^2$ is to compare the performance of the model at hand. To illustrate the problem, consider an artificial example. Asset returns follow a simple static factor model $r_t = B f_t + \epsilon_t$, with $B \in \mathbb{R}^{N \times K}$. The estimated factor returns are $\hat{f}_t = (B'B)^{-1} B' r_t$. We build a new factor model, where the estimated factor returns $\tilde{f}_t$ are a sequence of random vectors defined as follows:
$$ \tilde{f}_{1,t} = 0 \quad \dots \quad 0 $$
$$ \tilde{f}_{2,t} = 0 \quad \dots \quad 0 $$
$$ C_t = \begin{pmatrix} \tilde{f}_{1,t} & 0 & \dots & 0 \\ 0 & \tilde{f}_{2,t} & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \tilde{f}_{K,t} \end{pmatrix} $$
where $C_k^T C_k = C_k C_k^T = I_N$.
After some rote calculations, we find that the estimated factor returns from this model are
$$ \hat{\mathbf{f}}_t = (\mathbf{B}_t' \mathbf{B}_t)^{-1} \mathbf{B}_t' \mathbf{r}_t = C_t \mathbf{f}_t $$
The coefficient of determination is unchanged.
$$ R^2(\tilde{\mathbf{B}}_t) = 1 - \frac{\sum_{t=1}^T \| \mathbf{r}_t - (\mathbf{L}_t \mathbf{C}_t \mathbf{B}_t \mathbf{C}_t^{-1}) \mathbf{B}_t' \mathbf{r}_t \|^2}{\sum_{t=1}^T \| \mathbf{r}_t \|^2} $$
$$ = 1 - \frac{\sum_{t=1}^T \| (I_N - \mathbf{L}_t \mathbf{B}_t \mathbf{B}_t') \mathbf{r}_t \|^2}{\sum_{t=1}^T \| \mathbf{r}_t \|^2} $$
$$ = R^2(\mathbf{B}) $$
However, the estimated factor covariance matrix of the rotated model is
$$ [\hat{\Omega}_f]_{k,j} = \frac{1}{T} \sum_{t=1}^T \hat{f}_{k,t} \hat{f}_{j,t} $$
$$ \rightarrow \begin{cases} [\Omega_f]_{k,j} & \text{if } i=j \\ 0 & \text{if } i \ne j \end{cases} $$
When $N \gg K$, the average of $r_{it}^2$ is $\approx \sigma_{i\epsilon}^2$, i.e., it is identical to the variance of the original non-rotated model. When $N \approx K$, the sum approaches 0 because of the independence of $f_{k,t}$ and $f_{j,t}$, ($i \ne j$); and $E[f_{k,t} f_{j,t}] = E[f_{k,t}] E[f_{j,t}] = 0$.
The limitations of $R^2$ have led to the effect of decorrelating the estimated factor returns. To see how, we have here the effect of decorrelating the estimated factor returns. In summary we have two models, one is the true model, the other is a rotated model with reestimations in every period. The two models are indistinguishable in terms of $R^2$. The estimated factor returns are very different. The two models are also identical. However, the second model has a very different—and incorrect—covariance matrix. Expert modelers are aware of the shortcomings of $R^2$ for factor modeling, and resort to heuristics to confirm the correctness of the cross-sectional regression. One such heuristic is that the factor returns of a model which achieve a high $R^2$ should have significance criterion, i.e., the regression coefficient corresponding to a factor has an absolute t-score greater than two. Another natural check is on the realized Sharpe Ratios of each factor return. These tests confirm that $R^2$ is inadequate for factor model selection.[5] The recommendation is to rely on the tests we presented earlier in this chapter for risk model performance, and on Chapter 8 for testing risk-adjusted performance.

---

**5.4 Appendix**

**5.4.1 Proof for Minimum Variance Portfolios**

**Theorem 5.1:**
(Engle and Colacito, 2006) Let $\hat{\Omega}_k \in \mathbb{R}^{N \times N}$ be a candidate covariance matrix and $\Omega_t$ be the true covariance matrix. Let $\hat{\mathbf{w}}_k = \hat{\Omega}_k^{-1} \mathbf{1}_N$, and solve the risk minimization problem
$$ \min_{\mathbf{w}} \mathbf{w}' \hat{\Omega}_k \mathbf{w} $$
$$ \text{s.t. } \mathbf{1}_N' \mathbf{w} = 1 $$
and let $\mathbf{w}_k(\hat{\Omega}_k)$ be its solution. Then the realized variance of the portfolio $\text{var}(\mathbf{w}_k(\hat{\Omega}_k), \Omega_t)$ is greater than that of $\mathbf{w}_k(\Omega_t)$, and the two are identical if and only if $\hat{\Omega}_k = \Omega_t$.

Proof:
The solution of Problem (5.1) is
$$ \mathbf{w}(\hat{\Omega}_k) = (\mathbf{1}_N' \hat{\Omega}_k^{-1} \mathbf{1}_N)^{-1} \hat{\Omega}_k^{-1} \mathbf{1}_N $$
The ratio between realized variance of the portfolios constructed on $\hat{\Omega}_k$ and on $\Omega_t$ is
$$ \frac{\text{var}(\mathbf{w}(\hat{\Omega}_k), \Omega_t)}{\text{var}(\mathbf{w}(\Omega_t), \Omega_t)} = \frac{\mathbf{1}_N' \hat{\Omega}_k^{-1} \Omega_t \hat{\Omega}_k^{-1} \mathbf{1}_N}{(\mathbf{1}_N' \hat{\Omega}_k^{-1} \mathbf{1}_N)^2} \frac{(\mathbf{1}_N' \Omega_t^{-1} \mathbf{1}_N)}{\mathbf{1}_N' \Omega_t^{-1} \Omega_t \Omega_t^{-1} \mathbf{1}_N} $$
One can readily verify that if $\hat{\Omega}_k = \Omega_t$, the ratio is one. Let $\Omega_t^{1/2}$ be the SVD of $\Omega_t = U S U'$. Let $\mathbf{H} = \Omega_t^{-1/2} \hat{\Omega}_k \Omega_t^{-1/2}$.
Then we rewrite the variance ratio as
$$ \frac{\text{var}(\mathbf{w}(\hat{\Omega}_k), \Omega_t)}{\text{var}(\mathbf{w}(\Omega_t), \Omega_t)} = \frac{\mathbf{x}' \mathbf{H}^{-1} \mathbf{x}}{(\mathbf{x}' \mathbf{H}^{-1} \mathbf{x})^2} \frac{(\mathbf{x}' \mathbf{x})}{1} $$
Consider now the SVD of $\mathbf{H} = V D V'$ and define $\mathbf{y} = V' \mathbf{x}$. We have
$$ \frac{\text{var}(\mathbf{w}(\hat{\Omega}_k), \Omega_t)}{\text{var}(\mathbf{w}(\Omega_t), \Omega_t)} = \frac{(\sum_i y_i^2/d_i)}{(\sum_i y_i^2/d_i)^2} (\sum_i y_i^2) $$
The term on the RHS can be interpreted as $E[1/U]$, where $U$ is a random variable taking value $d_i$ in state $i$ with probability $p_i = y_i^2 / \sum_j y_j^2$. By Jensen’s inequality, $E[1/U] \ge 1/E[U]$, and the result follows.

**The Takeaways**

1.  When evaluating a factor model, we are concerned with three dimensions of performance:
    *   Accuracy of the covariance matrix, since the portfolio’s volatility prediction accuracy depends on it.
    *   Accuracy of the precision matrix, since it is used in mean-variance portfolio optimization.
    *   Ancillary performance metrics like turnover and beta prediction, since they are important in trading applications.
2.  Two principled metrics for volatility estimation are QLIKE and MSE.
3.  They can be extended to the multivariate case, allowing for comparison of models across production strategies (best-case, worst-case, etc.) focusing on realistic portfolio scenarios.
4.  For precision matrix accuracy, we use Minimum Variance Portfolios: construct portfolios satisfying a linear constraint. A lower realized variance indicates a better model.
5.  Another metric is the Mahalanobis distance test. It’s portfolio-independent.
6.  Ancillary metrics are as follows:
    *   Turnover: Considers the frequency of model updates and potential transaction costs, with measures such as FMP turnover.
    *   Beta Accuracy: Compares predicted versus realized betas for portfolios (e.g., thematic, factor-mimicking) to ensure thematic, factor-mimicking to ensure consistency.
7.  We caution against the use of $R^2$ for factor model evaluation as it can be manipulated and may not reflect true predictive performance.

**Notes**

1.  [1] The Bias statistic is defined as $T^{-1} \sum_{t=1}^T (\hat{\sigma}_{it}^2 - r_{it}^2)$. See Connor et al. (2010).
2.  [2] A note of caution regarding the computation of QDIST in Equation (5.2): a numerically stable way to compute $\log |\mathbf{A}|$ is to compute the SVD of the argument and exploit the fact that
    $$ |\mathbf{A}| = |U S V^T| = |S| $$
    , so
    $$ \log |\mathbf{A}| = \log |S| = \sum_i \log s_i $$
3.  [3] To prove this, note that the vector $\mathbf{z}$ can be generated by $\mathbf{r} = \Omega^{1/2} \mathbf{z}$, where $\mathbf{z} \sim N(0, I_N)$. Therefore,
    $$ \mathbf{r}' \Omega^{-1} \mathbf{r} \sim \mathbf{z}' \Omega^{-1/2} \Omega \Omega^{-1/2} \mathbf{z} = \mathbf{z}'\mathbf{z} \sim \chi_N^2 $$
4.  [4] I should note that some researchers do model time variation of execution costs by including trading volume. But these approaches are out of the scope of this book.
5.  [5] We ignore corrections of $R^2$ for the “degrees of freedom” and alternative measures of quality of fit, like Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) (e.g., Hastie et al. (2008)), since these corrections pertain to in-sample fit.
6.  [6] For a more critical perspective on $R^2$ used as a metric for time-series model, see Lewellen et al. (2010).

```

Okay, here is the Markdown compilation for Chapter 6.
     Everand

