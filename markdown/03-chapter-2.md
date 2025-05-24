**Chapter 2**
**Univariate Returns**

**The Questions**

1.  What are the definitions and types of returns, including dividend-adjusted and excess returns?
2.  What are the “stylized facts” of stock returns, and why are they important?
3.  How do we estimate prices and returns while accounting for market microstructure effects?
4.  What are Conditional Heteroskedastic Models (CHMs), and how do they model volatility?
5.  How does the GARCH(1,1) model capture the stylized facts of returns?
6.  What is realized volatility and how is it estimated using high-frequency data?
7.  How can state-space models and the Kalman filter be used for variance estimation?
8.  How does GARCH(1,1) relate to Exponentially Weighted Moving Averages (EWMA) in volatility estimation?

We begin with models of univariate returns for two reasons. First, single-asset returns are the basic constituents of portfolios. We cannot hope to understand portfolio returns if we do not have a solid understanding of the returns of single assets. It is necessary to summarize the salient empirical properties of stock returns and the most common processes employed to model them, specifically to model volatility effectively. Second, these models have general applicability and are even more useful when we combine them with models of multiple assets. We introduce GARCH and exponentially weighted moving averages as essential tools for the estimation of volatility. In the process, I introduce models that justify their use. Exponential moving averages find their motivation in linear state-space models, while GARCH is an instance of a non-linear state-space model. These models will be your friends for life. The chapter is organized as follows. First, we define returns and their basic properties. Then, we introduce some “stylized facts” (empirical features of returns that are ubiquitous and relevant to risk management). As part of basic volatility modeling, we will introduce GARCH models and realized volatility models. Given that these topics are vast, I make no attempt to be exhaustive; the goal here is to present the essentials, their associated insights, and to provide a working knowledge of the models. Finally, I will touch on the state-space models for variance estimation.

---

**2.1 Returns**

**2.1.1 Definitions**

We have a set of $N$ assets and a currency, also called the numeraire. Throughout this book, the numeraire is the U.S. dollar. The price of an asset $i$ is the amount of numeraire necessary to acquire one unit of these assets. In defining returns, we buy today the equivalent of 1 unit of currency for asset $i$. We denote the value of the asset tomorrow by some future price $P_{i}(t+1)$. An equivalent way to define returns is from the closing prices of security $i$ on days $t$ and $t-1$, which we denote $P_i(t)$ and $P_i(t-1)$ respectively. The return is defined as

$$ r_i(t) = \frac{P_i(t) - P_i(t-1)}{P_i(t-1)} $$

We extend this definition to the case in which the security pays a dividend. The holder of the asset receives an amount $D_i(t)$. The dividend-adjusted return is defined as

$$ r_i(t) = \frac{P_i(t) + D_i(t) - P_i(t-1)}{P_i(t-1)} $$

In a universe of $N$ assets, the vector of daily returns between times $t-1$ and $t$ is denoted
$$ \mathbf{r}(t) = (r_1(t), \dots, r_N(t))' $$
For a portfolio $p$, let $w_{ip}$ be the monetary amount invested in asset $i$. Thus, a single period return is given by the change in the value of the portfolio. The number of shares owned in asset $i$ is calculated as $w_{ip} / P_i(0)$. The value of the portfolio at period $t$ is
$$ \sum_i w_{ip} (P_i(t) / P_i(0)) P_i(1) $$
, and the change in value is
$$ w_p \frac{P(1) - P(0)}{P(0)} = \sum_i w_{ip} $$
In vector form, this equals $w_p'$. The volatility of a random return is its standard deviation:
$$ \sigma_i := \sqrt{E[(r_i - E(r_i))^2]} $$

The variance is the square of the volatility. Occasionally, when the approximation $P(t) \approx E[P(t)]$ holds, we will approximate the volatility by the second moment of the return, $E[r_i^2]$. The compounded return over $T$ periods of investing one unit of currency in asset $i$, which has been invested in a security yielding returns $r_i(t)$, is:

$$ r_i(1:T) = \frac{P_i(T)}{P_i(0)} - 1 $$
$$ = \frac{P_i(T)}{P_i(T-1)} \frac{P_i(T-1)}{P_i(T-2)} \dots \frac{P_i(1)}{P_i(0)} - 1 $$
$$ = \prod_{t=1}^T (1+r_i(t)) - 1 $$

**2.1.2 Excess Returns**

In the rest of the book, we will not use security returns, but excess returns: the risk-free rate $r_f$. For example, we model daily returns, the risk-free rate $r_f$ is the interest rate paid by the investor for borrowing cash over the same period, or paid to the investor for cash held in their account. If we hold a security, we pay interest on the cash we used to buy the security. If we short a security, we receive cash. Cash is an asset with a security, but a value in the future that is much less uncertain than other modeling purposes, negligible volatility than the other risky assets. We borrow or lend an amount equal to the Fair Market Value (FMV) of our portfolio, i.e., the sum of the values of each position. The return of a portfolio is
$$ \sum_i w_{ip} r_i - \sum_i w_{ip} r_f = \sum_i w_{ip} (r_i - r_f) $$
The formula shows us that the risk-free rate is subtracted from the portfolio and provides a natural interpretation of security returns as returns in excess of a rate received in the absence of investing. In the United States, the reference rate is a reference overnight lending rate, like the Secured Overnight Financing Rate (SOFR).[2]

**2.1.3 Log Returns**

If $r_i(t)$ follows a multivariate Gaussian distribution, then so does the portfolio’s P&L. The variance of the portfolio can be computed by just using two pieces of information: the portfolio weights and the covariance matrix of the returns.

The question of whether net returns are Gaussian is an empirical one. We at least know that the distribution of daily returns is not Gaussian. Log returns are additive over time. However, there are not easily tractable time-series available. For example, consider the compound return over period $T$, $\prod_{t=1}^T (1+r_i(t))$. If $r_i(t)$ are normally distributed, the cumulative total return is not normally distributed, and its distribution rapidly diverges from the normal distribution. The variance of the cumulative returns is not a simple function of the single-period variances.

On the other side, log returns are attractive because they are additive. Let
$$ \tilde{r}_i(t) := \log(1+r_i(t)) $$
Then, the log of the compound return is equal to the sum of the log returns in the single periods. If the log returns are normal, then the sum of the continuously compounded returns are also normally distributed, and the variance of the sum is equal to the sum of the variances. We can reconcile the two views of returns—raw and log—if the approximation
$$ \log(x) \approx x-1 + o(|x-1|^2) $$
is sufficiently accurate, i.e., if net returns are small. In this case, we can make the approximation $\tilde{r}_i(t) \approx r_i(t)$, which is sufficiently accurate provided the returns are not too large.

A common approximation for the compounded net return of an asset over time is given by
$$ \prod_{t=1}^T (1+\tilde{r}_i(t)) - 1 = \exp\left(\sum_{t=1}^T \tilde{r}_i(t)\right) - 1 $$
$$ \approx 1 + \sum_{t=1}^T \tilde{r}_i(t) - 1 $$
$$ = \sum_{t=1}^T \tilde{r}_i(t) $$
Above we rely on the accuracy of the approximation. For example, comparing the estimates of models developed using $r_i(t)$ and $\tilde{r}_i(t)$. The approximation is in general optimistic (see Exercise 2.1). When the assets are equities, the approximation is usually considered adequate for daily internal measurements or shorter. For long time intervals (e.g., yearly) or very volatile returns, the approximation is poor.

**2.1.4 Estimating Prices and Returns**

To estimate a security’s price, $P(t)$, we need prices. Prices, however, depend crucially on the way a market is organized. OTC markets (Harris, 2003) differ from exchanges that use limit-order books (Bouchaud et al., 2018). Within a single exchange, the trading mechanisms can change over the course of the day, with auctions often taking place at the beginning and at the close of the trading day. As a result of market design, the price of a security is not unique. There are many prices. The most common example of ambiguity is the bid-ask spread. In limit-order books, the best bid (ask) price is the price attribute (the “bidding” price per share the buyer is willing to pay) and a quantity. Similarly, the sell orders have a price attribute, or “asking prices,” and a quantity. Buying orders have higher asking prices, and the difference is called the bid-ask spread. This spread is a measure of liquidity. In order for a transaction to occur, a buy order or a sell order must cross the spread; either event can occur. As a result, the transaction price will be either at the top or the bottom of the bid-ask spread interval. Successive transactions will have different prices, even if the value of the asset is constant. Among transactions, the bid-ask spread implies that there is not the only source of randomness in prices. Observed prices can differ by exchanges, and the selection of price by timestamp depends on the choice of data integration. Then, there may be slight measurement errors. It is important to consider the outset the fact that prices are imperfectly observed, and that this imperfection is a source of noise in the returns. This noise is, perhaps, the simplest model is the Roll model for asset prices (Roll, 1984). In this model, $m_t$ is the “true” price. The true asset evolves as an arithmetic random walk, and we imperfectly observe the price $P_t$. In formulas:
$$ m_{t+1} = m_t + \sigma_m \epsilon_{t+1} \quad \text{(evolution)} $$
$$ P_{t+1} = m_{t+1} + \sigma_\eta \eta_{t+1} \quad \text{(observation)} $$
where $\epsilon_{t+1}, \eta_{t+1}$ are random variables, independently distributed (specifically, and from each other) according to a standard normal distribution.

Before we try to estimate prices, the model presents an immediate and testable consequence: consecutive price differences are negatively correlated. The price difference is
$$ \Delta P_{t+1} = P_{t+1} - P_t = (m_{t+1} - m_t) + \sigma_\eta (\eta_{t+1} - \eta_t) $$
which is zero in expectation. However,
$$ E[\Delta P_{t+1} \Delta P_t] = -\sigma_\eta^2 $$
The lag-one autocorrelation, $E[\Delta P_{t+1} \Delta P_t] / \text{Var}(\Delta P_t)$ can also be used to estimate the measurement error. Equation (2.2) can also be used to estimate the measurement error. The presence of large non-zero autocorrelations beyond lag one may point to model inadequacy, in the sense that there are actual long-term dependencies in prices. The true price $m_t$ is not observed. We can estimate it, however, by using the Kalman filter. The Kalman filter is described in the Appendix, Section 2.4, and specifically in Example 2.2 of Section 2.4.2. The estimator is given by
$$ \hat{m}_{t+1|t} = (1-K) \hat{m}_{t|t-1} + K P_t $$
where the optimal formula $K \in (0,1)$ is given in the Appendix. The smaller the ratio $\sigma_m^2 / \sigma_\eta^2$, the higher the $K$, which makes sense: we do not need to average observations if the price observations are accurate. The value of the estimate is an exponentially weighted average of past prices. The value of the estimate $m_t$ is the price in the measurement period. If we want the daily closing price, for example, we may want to use a weighted average of 5-minute interval prices in the preceding intervals. There is a caveat, however. Suppose we have estimates $\hat{m}_n$ and we use these estimates to compute returns
$$ \hat{r}_n := \hat{m}_n / \hat{m}_{n-1} - 1 $$
Because we employ the same observed prices $P_n$ (both in $\hat{m}_n$ and $\hat{m}_{n-1}$), the estimates are positively correlated. One should always check that $(1-K)^2 < 0.5$, otherwise this spurious correlation.

[Image: ACF plots for (a) IBM, (b) MSFT, (c) XOM, (d) AAPL, (e) SPY, (f) XLK. Figure 2.1: Autocorrelation plots of daily log return series (symbols). The horizontal dashed lines delimit the 95% confidence interval for the autocorrelations.]

**2.1.5 Stylized Facts**

Before building the house, we need to look at the bricks, namely, the statistical properties of the single-stock returns. Below we list some “stylized facts” about stock returns and discuss their relevance to risk modeling and management. Returns have a lower signal-to-noise ratio, and their statistical properties of
$$ \tilde{r}(t) = \log(1+r(t)) $$
We focus on the properties of $\tilde{r}(t)$ because they are additive. The uncentered volatility of the log returns is $\tilde{\sigma}(t) = \sqrt{E[\tilde{r}^2(t)]}$. These properties are well known (Cont, 2001; Taylor, 1986; Tsay, 2010; Ruffini-Crouzet et al. (2019)).

1.  Absence of autocorrelations. Logged stock returns are uncorrelated unless you observe prices and returns at time scales at which the market microstructure becomes relevant (say, intraday). See Figure 2.1.
2.  Heavy tails. The unconditional distribution of returns shows heavy tail behavior. That is, the probability of large returns is much larger than the probability of large returns under a normal distribution. This means that the probability of a large return is much higher than what would be consistent with any “thin-tailed” distribution with infinite moments. Examples of sample kurtosis are in Table 2.1. The conditional (say, conditional on the entire past history up to time $t-1$) distribution of returns may have “heavy tails” behavior as well, but with lighter tails than the unconditional one.

**TABLE 2.1**
Sample skewness and kurtosis of daily log returns and “hold” (italic p equals 0.01) confidence intervals estimated using non-parametric bootstrap with replacement (5000 variations). Range: 1/1/2000–12/31/2017.

| Stock | Skewness Left | Skewness Right | Kurtosis Left | Kurtosis Right |
| :---- | :------------ | :------------- | :------------ | :------------- |
| AAPL  | -0.2          | 0.2            | 5.7           | 7.8            |
| IBM   | -0.5          | 0.2            | 3.6           | 3.6            |
| XOM   | -0.4          | 0.3            | 7.1           | 8.7            |
| MSFT  | -0.3          | 0.2            | 5.8           | 8.1            |
| WAT   | -1.0          | -0.3           | 12.8          | 40.1           |
| SPY   | -0.3          | -0.7           | 8.6           | 11.4           |
| XLK   | -0.3          | 0.1            | 6.5           | 16.0           |

Regarding heavy tails for asset returns, we restrict our attention to power-tailed distributions. Then the complement of the cumulative distribution function follows a power law:
$$ P(Z > x) = P(|Z| > x) \approx Cx^{-\alpha} $$
for large $x$, where $Z$ is the random variable (e.g., daily log returns), $C > 0$ is a constant, and $\alpha > 0$ is the tail index. Compare this to a Gaussian return: $P(Z > x)$ is the cumulative distribution function (CDF) of the Gaussian. There is a common approximation (Wasserman, 2004) for the tail probability is
(2.4)
$$ \frac{1}{\sqrt{2\pi}} \frac{1}{x} e^{-x^2/2} \left(1 - \frac{1}{x^2}\right) \le P(Z>x) \le \frac{1}{\sqrt{2\pi}} \frac{1}{x} e^{-x^2/2}, x > 0 $$
The right-side inequality can be used to bound the error in the left tail and the symmetric inequality of the right tail for $x > K$ where $K \approx 0.025$.
(2.5)
$$ P(x) \le \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \left(1 - \frac{1}{x^2}\right) \left(1 - \frac{1}{2x^2} (1 - \frac{1}{x^2})\right)^{-1} $$
(2.6)
$$ P(x) \ge \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \left(1 - \frac{1}{x^2}\right) \sqrt{1 - \frac{2}{x^2} \log(2\sqrt{2\pi} \cdot 1)} $$
The approximation is quite accurate. Let the Cumulative Density Function $F(x) = 1 - P(x)$. We have the bounds
$$ \frac{1}{\sqrt{2\pi}(1+x)} e^{-x^2/2} \le P(x) \le \frac{1}{\sqrt{2\pi}x} e^{-x^2/2} \quad (x > 0) $$
$$ (1-d) < K < 0.025 $$
A Gaussian random variable has finite moments of any order. A power-law random variable with exponent $\alpha$ has finite moments only up to $k < \alpha$. A Gaussian random variable has finite moments of any order. A power-law random variable with exponent $\alpha$ has finite moments only up to $k < \alpha$. If $\alpha \in (2,4]$, the unconditional distribution of log returns has heavy tails. It is still not settled what the exponent $\alpha$ associated with the distribution. It seems, however, that $\alpha \in (2,4]$. This is important for several practical purposes. A sufficient condition for the applicability of the Central Limit Theorem says that if we have $n$ iid random variables with mean $\mu$ and variance $\sigma^2 < \infty$, then their sum converges in distribution to a Gaussian random variable with mean $n\mu$ and variance $n\sigma^2$. The Theorem allows us to establish the asymptotic distribution of estimators. However, if $\alpha \le 2$, then the variance of the returns is infinite. This seems to be the case. However, a related question is whether the conditional distribution is heavy-tailed, and it is possible to model returns as a process with conditional Gaussian returns.

3.  Autocorrelation of absolute returns and second moments. The time series of $|r(t)|$ or $r^2(t)$ shows positive, significant, and slowly decaying autocorrelation functions (the “volatility clustering” effect in the literature (Taylor, 1986; Granger and Ding, 1995)). Thus, large realized return movements persist over time, and so do small realized return movements. This phenomenon is sometimes termed volatility clustering.
4.  Aggregational Gaussianity. At longer time scales (say, weekly or monthly returns, as opposed to daily or intraday returns), the distribution of log returns becomes closer to a Gaussian distribution. Reality, is in stark contrast with simple models of univariate price dynamics like the geometric Brownian motion process or other simple discrete-time models.
    $$ dP(t) = \mu P(t)dt + \sigma P(t)dW(t) $$
    (price change) (drift) (noise)
    This model predicts Gaussian, independent log returns, which are inconsistent with the empirical evidence. First, returns show little serial autocorrelation. This does not mean that returns are independent, nor that returns are unpredictable based on their history. Some authors interpret the absence of autocorrelation as evidence that returns are not serially predictable. Regarding the latter, returns are predictable. This is not only an article of faith of active investors, who usually do a terrible job at it, but also a relatively uncontroversial empirical finding among academics.[7] Nevertheless, even though they are predictable, they are not so linearly predictable.

... heavy-tailed unconditional ones. This family, the Conditional Heteroskedastic Models (CHM), is rich and the subject of the following subsection. We won’t cover models with jumps, which are also popular. We also won’t cover models like Levy processes and FARIMA models. No model covers all the empirical features observed in stock returns. GARCH models (and mixture models in general) have the benefit of being easy to interpret, simulate, and estimate.

---

**2.2 Conditional Heteroskedastic Models**

This family of models was first proposed in the early 1980s by Engle (1982), Engle and Bollerslev (1986). By the mid-1990s they had been generalized and applied in several economic domains.[8] They are extensively covered in any econometrics book.

The most popular and studied model in this family is the GARCH(1,1) model. It has good empirical properties, its theoretical properties have been characterized, and it can be estimated efficiently. It also conveys the gist of the large set of models in this family. The GARCH(1,1) model is a part of the class of stochastic process. The form of the GARCH(1,1) is:
(2.8)
$$ h_t^2 = \alpha_0 + \alpha_1 r_{t-1}^2 + \beta_1 h_{t-1}^2 $$
(2.9)
$$ (\alpha_0 > 0, \alpha_1 \ge 0, \beta_1 \ge 0, \alpha_1 + \beta_1 < 1) $$
In the equations above, $h_t^2$ is the volatility at time $t$ (and $h_{t-1}^2$ is the variance) and it is conditional on the value of $r_{t-1}^2$. The parameters $\alpha_0, \alpha_1, \beta_1$ are constants that need to be estimated. The sum of the parameters $\alpha_1 + \beta_1$ determines by Equation (2.8). To gain some intuition, let us look at the second equation of the GARCH process when we remove the term $r_{t-1}^2$. The equation can be rewritten as:
(2.10)
$$ h_t^2 = \alpha_0 + \beta_1 h_{t-1}^2 $$
$$ h_t^2 - \beta_1 h_{t-1}^2 = (h_t^2 - h_{t-1}^2) + (1-\beta_1) h_{t-1}^2 $$
where
$$ h_t^2 = \frac{\alpha_0}{1-\beta_1} $$
This value $h^2$ converges to $h_t^2$ at a geometric rate, so long as $0 < \beta_1 < 1$. The term $\alpha_1 r_{t-1}^2$ in Equation (2.8) increases the probability of large squared returns in the following period, giving rise to a rich dynamic behavior. The increase in volatility can not continue unabated, because the term $(1-\alpha_1-\beta_1)h_{t-1}^2$ will dampen variances that are much greater than the “equilibrium level” $h^2$. This can be seen through substitution in the second equation of the model.

(2.11)
$$ h_t^2 = h^2 + \alpha_1 \sum_{i=1}^\infty \beta_1^{i-1} (r_{t-i}^2 - h^2) $$
One could replace the true values $h^2, \alpha_1, \beta_1$ with their estimates, and interpret the formula as saying that the current variance an exponential moving average of non-iid squared returns, since they are modulated by $h^2$ in light of Equation (2.11).

**2.2.1 GARCH(1,1) and Return Stylized Facts**

The GARCH(1,1) model improves on the distributional properties of conditional returns by making them closer to the Gaussian distribution, see Figure 2.2. How does the GARCH(1,1) model stack up against the stylized facts?

[Image: Figure 2.2 Quantile-Quantile plot for daily log returns (light gray dots) and GARCH(1,1) (dark gray dots) for (a) AAPL, (b) IBM, (c) XOM, (d) MSFT, (e) SPY, (f) XLK. The theoretical quantiles are from a $t$-distribution with $\nu$ degrees of freedom ($t_\nu(0,0001 \cdot 1.8)$ for XLK).]

1.  Absence of autocorrelations. This property is satisfied (not hard to verify directly).
2.  Heavy tails. The unconditional returns are leptokurtic (Cont, 2001).

In GARCH models (Mikosch and Stărică, 2000), the distribution of the unconditional returns is heavy tailed. In GARCH(1,1) models estimated on daily stock returns, the tail index $\alpha$ of the unconditional returns is often close to a Gaussian distribution; see Table 2.2. The tail indices of $r_t^2$ are also higher than those of $r_t^2$; see Table 2.3. These are desirable properties of GARCH processes.

**TABLE 2.2**
Distances between the theoretical normal distribution and the empirical distribution of the residuals of GARCH(1,1). The distances are produced for all stocks, with the values reported in the table for the two proxies for the market (SPY) and the technology sector (XLK). N.B.: We use the Kolmogorov–Smirnov distance. Background on this statistic can be found from Theil (1971) and Conover (1971, Ch. 6).

| Stock Uncond. | GARCH(1,1) Left parenthesis | GARCH(1,1) Left parenthesis epsilon sub t right parenthesis |
| :------------ | :-------------------------- | :------------------------------------------------------- |
| AAPL          | 0.039                       | 0.041                                                    |
| IBM           | 0.079                       | 0.047                                                    |
| XOM           | 0.060                       | 0.060                                                    |
| MSFT          | 0.061                       | 0.060                                                    |
| WAT           | 0.100                       | 0.091                                                    |
| SPY           | 0.098                       | 0.040                                                    |
| XLK           | 0.091                       | 0.043                                                    |

**TABLE 2.3**
Estimated tail index for left and right tail of probability density
*hold-italic p equals 0.05*, proportional to *hold-italic alpha*, a super-negative *hold-italic alpha*.
We use the Maximum Likelihood Estimator (MLE) *alpha hat* equals one plus *hold-italic alpha* times left square bracket *n* times sum from *i* equals one to *n* of log of *hold-italic x sub i* over *hold-italic x sub min* right square bracket *hold-italic alpha* sub *n* right square bracket super negative one
where *hold-italic x sub min* is the minimum value of *hold-italic x* (Clauset et al. 2009). The value of *hold-italic x sub min* is set to -2.5% and 2.5%, respectively. Estimates of *hold-italic alpha* increase stability for the two indexes SPY and XLK.

| Stock Uncond. | Left Tail GARCH(1,1) *hold-italic alpha* | Right Tail GARCH(1,1) *hold-italic alpha* |
| :------------ | :--------------------------------------- | :---------------------------------------- |
| AAPL          | 4.8                                      | 4.8                                       |
| IBM           | 3.8                                      | 4.2                                       |
| XOM           | 4.0                                      | 4.6                                       |
| MSFT          | 3.9                                      | 3.8                                       |
| WAT           | 3.4                                      | 4.1                                       |
| SPY           | 3.0                                      | 4.2                                       |
| XLK           | 3.0                                      | 4.4                                       |

3.  Autocorrelation of absolute and squared returns. The ACF for GARCH(1,1) is positive for both absolute and squared returns; for squared returns, it has the form (He and Teräsvirta, 1999; Ruppert and Matteson, 2015)
    $$ \rho_k = \begin{cases} \frac{\alpha_1 (1-\alpha_1 \beta_1 - \beta_1^2)}{1-2\alpha_1 \beta_1 - \beta_1^2} & \text{if } n=1 \\ \rho_1 (\alpha_1 + \beta_1)^{k-1} & \text{if } n > 1 \end{cases} $$
    However, the decay for squared returns and squared residuals is too slow. The model appears that the autocorrelation predicted by the model for a given observed horizon level is too high compared to that observed in practice. See Teräsvirta (2009a).
4.  Aggregational Gaussianity. A GARCH process aggregated over longer horizons is GARCH with different parameters and still heavy-tailed (Drost and Nijman, 1993). This is consistent with the autocorrelation issue above: roughly, GARCH has a “long-term memory” that is too long, and not completely in agreement with empirical data.

Summing up, some but not all of the stylized facts about log returns are captured by GARCH(1,1).

**2.2.2 GARCH as Random Recursive Equations**

We now look at GARCH(1,1) through different modeling approaches. First, we could reformulate it as a random-iterated function. Rewrite Equation (2.8) as
$$ h_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 h_{t-1}^2 + \beta_1 h_{t-1}^2 $$
Set
$$ a_t := \beta_1 + \alpha_1 \epsilon_t^2, \quad b_t := \alpha_0 $$
The random variables $a_t, b_t$ are iid. Then
$$ h_t^2 = a_{t-1} h_{t-1}^2 + b_{t-1} $$
This formulation shows that the process is a linear state-space model and also that it is governed by an autoregressive moving average with random coefficients. By recursive iteration (Embrechts, 2006), we can rewrite the equations as
$$ h_t^2 = (\prod_{j=0}^{k-1} a_{t-j}) h_{t-k}^2 + \sum_{j=0}^{k-1} (\prod_{i=0}^{j-1} a_{t-i}) b_{t-j-1} $$
This allows us to study the process using the toolkit of random recursive equations.[12]
The distribution of $\log a_t$ plays the essential role in the convergence of the process. Nelson (1990) characterizes the conditions for convergence. In the case $\alpha_0 > 0$:
1.  If $E[\log a_t] < 0$, then $h_t^2 \xrightarrow{d} h^2$, where $h^2$ is a non-degenerate distribution.
2.  If $E[\log a_t] \ge 0$, then $h_t^2 \xrightarrow{p} \infty$ as $t \to \infty$.
3.  If $E[\log a_t] = 0$, then $h_t^2$ converges to a non-degenerate distribution with support on $[h_0^2/(1-\beta_1), \infty)$.
The kurtosis of the process is
$$ \frac{3(1-(\alpha_1+\beta_1)^2)}{1-(\alpha_1+\beta_1)^2 - 2\alpha_1^2} - 3 = \frac{6\alpha_1^2}{1-3\alpha_1^2 - 2\alpha_1\beta_1 - \beta_1^2} > 0 $$
as long as $\alpha_1 > 0$, so that the process is leptokurtic. How about skewness? The unconditional returns are not skewed, because
$$ E(r_\infty - E(r_\infty))^3 = E(h_\infty^3) E(\epsilon_\infty^3) = 0 $$
Finally, we point out that not only are the unconditional returns leptokurtic, but they do in fact have Pareto tails, provided the process is stationary: $P(r_t > x) \sim C x^{-\kappa}$ for some $\kappa > 0$, see Mikosch and Stărică (2000); Basrak, Buczkowski et al. (2008).

**2.2.3 GARCH(1,1) Estimation**

Although GARCH models are highly general, the vast majority of CHM applications use GARCH(1,1). This is not an accident. GARCH(1,1) is easy to estimate, and its properties are well understood. Generalization to higher-order models is straightforward. Define $\theta = (\alpha_0, \alpha_1, \beta_1)$, and let $f$ be the log-density function of the standard normal distribution.
$$ \tilde{r}_t = h_t \epsilon_t $$
$$ h_t^2 = \alpha_0 + \alpha_1 \tilde{r}_{t-1}^2 + \beta_1 h_{t-1}^2 $$
where $h_0^2$ is defined by the right-hand side in Equation (2.10). By repeated substitution, we can express the unobserved variance $h_t^2$ as a function of the sequence $\tilde{r}_1, \dots, \tilde{r}_{t-1}$ and $\theta$. The log-likelihood of the sequence $\tilde{r}_1, \dots, \tilde{r}_T$ is
$$ L(\theta) = \sum_{t=1}^T f\left(\frac{\tilde{r}_t}{h_t(\tilde{r}_1, \dots, \tilde{r}_{t-1}, \theta)}\right) $$
We can then estimate the parameters $\theta$ of the model by maximizing the log-likelihood. As an example, consider the GARCH(1,1) model. The recursive equation for $h_t^2$ is given by Equation (2.11). We solve

$$ \min_\theta \sum_{t=1}^T \left( \log h_t^2(\theta) + \frac{\tilde{r}_t^2}{h_t^2(\theta)} \right) $$
s.t. $h_t^2 = (1-\beta_1)h^2 + \alpha_1 \sum_{i=1}^\infty \beta_1^{i-1} (\tilde{r}_{t-i}^2 - h^2)$, $i=1, \dots, T$

**2.2.4 Realized Volatility**

CHMs model the asset volatility as an (unobserved) state of the return stochastic process. Once we have an estimate of the volatility at time $t-1$ returns, the next arrival. An alternative route would be to estimate directly the volatility of the returns. This is the approach taken by realized volatility (RV) measures. This approach would not work if the time intervals at which we need the estimates are days, and we only have daily data. In recent years, tick-level price data have become widely available; indeed, order-book-level data are also available with high-frequency price data or order arrivals, trades, and cancellations. Realized volatility is computed non-parametrically, making it possible to estimate the volatility of daily returns by using these high-frequency data. Below we review some of the statistical properties of realized volatility (RV) measurements. The starting point is Equation (2.7), i.e., a diffusion process for the log price $p(t) = \log P(t)$
$$ dp = \alpha dt + \sigma dW $$
where $W(t)$ is a Brownian motion process, the term $\alpha dt$ is the drift, and $\sigma > 0$ is the volatility. We are interested in estimating $\sigma$. In practice, the drift is much smaller than the volatility ($|\alpha| \ll \sigma$). The quantity $\sigma/\sqrt{n}$ is termed the Sharpe Ratio and will figure prominently in the rest of the book. We observe the process in the interval $[0, T]$ and measure the state variable $p$ at intervals of length $T/n$. The measured returns are
$$ r(j) = p(j T/n) - p((j-1)T/n) $$
The RV is an unbiased estimator of $\sigma^2$.
$$ RV(n) = \frac{n}{T} \sum_{j=1}^n r(j)^2 \sim N(\sigma^2, \sigma^4/n) $$
The MLEs for drift and moments are
[Eq. 2.16]
$$ \hat{\alpha} = \frac{1}{T} \sum_{j=1}^n r(j) = \frac{1}{T} (p(T) - p(0)) $$
$$ \hat{\sigma}_n^2 = \frac{1}{T} \sum_{j=1}^n (r(j) - \hat{\alpha}/n)^2 $$
We also consider the uncentered estimator of the volatility $\hat{\sigma}_n^2$.
The first remarkable phenomenon is that the MLE for the drift, Equation (2.16), does not depend on the number of intermediate observations. Since our focus is on volatility, we assume $\alpha=0$. Then
$$ \text{Var}(\hat{\alpha}) = \text{Var}(p(T)-p(0)) = \sigma^2 T $$
and
$$ p(T) - p(0) \sim N(0, \sigma^2 T) $$
, so that $\text{Var}(\hat{\alpha}) = \sigma^2 T$. The estimation error does not depend on the number of intermediate observations $n$. To estimate the variance of $\hat{\sigma}_n^2$, we need a bit of gymnastics. The variance of $\hat{\sigma}_n^2$ is the sum of $n$ terms of a chi-square random variable with mean $\sigma^4/n$ and variance $2\sigma^8/n^2$.
(2.17)
$$ E[\hat{\sigma}_n^2] = \sigma^2 $$
(2.18)
$$ E[(\hat{\sigma}_n^2)^2] = \sigma^4 \left(1 + \frac{2}{n}\right) + \frac{\sigma^4}{n} \left(\frac{\alpha T}{n \sigma^2}\right)^2 $$
so that
(2.19)
$$ \text{Var}(\hat{\sigma}_n^2) = \frac{2\sigma^4}{n} \left(1 + \frac{2}{n}\right) + \frac{2\sigma^4}{n^2} \left(\frac{\alpha T}{n \sigma^2}\right)^2 $$
and
$$ E[\hat{\sigma}_n^2] = \sigma^2 + \frac{\sigma^2}{n} \quad \text{from Equation (2.17)} $$
$$ \text{Var}(\hat{\sigma}_n^2) = \frac{2\sigma^4}{n} \left(1 + \frac{4}{n}\right) + \frac{\sigma^4}{n^2} \quad \text{from Equation (2.19)} $$
The estimator $\hat{\sigma}_n^2$ has a small finite-sample bias and is asymptotically consistent.

**Insight 2.1: Estimating variance**
Based on Equations (2.17), (2.18), and (2.19), you can use uncentered returns for variance estimation, since the bias is inversely proportional to $n$, and the estimator is consistent.

Let us reflect on the steps we took. We discretized the interval over which the price process evolves into $n$ sub-intervals of length $T/n$. We assumed the price had no measurement error. We saw that the drift estimator is unbiased, but its variance does not depend on the discretization, $n$. We saw that the drift estimator is unbiased, but its variance does not depend on the discretization, $n$; there is no easy way to measure the drift, but they are noisy. Unfortunately, there is no easy way to measure the drift. It is estimated by returns, which have variance $\sigma^2 T$. We have used an uncentered estimator of the true variance $\sigma^2$. As the number of intervals approaches infinity, the estimator is unbiased, its variance decreases like $2\sigma^4/n$, which is good news in principle. We can use estimates to address daily volatility using very short time intervals, and we can combine these estimates to get an estimate of volatility over longer periods. If you need volatility estimates over a long time scale for your decisions (e.g., days), but have data over a shorter time scale (e.g., minutes), you do not have to throw a generative model like CHMs or others. What assumptions do not hold in this line of reasoning? There is a list of issues to consider:
1.  Intraday market microstructure. One source of noise is the bid-ask spread (Harris, 2003). When the buyer initiates a transaction, he pays the ask price. When the seller initiates it, he receives the bid price. There is an intrinsic error in the measurement of price, which is systematically biased toward the bid-ask spread. The model error by assuming that log returns are iid normal, when the noise terms $\eta_t$ are independent, identically distributed random variables.
2.  Another form of microstructure imperfection is thinly traded securities. If a stock trades less than once every 5 minutes on average, using 1-minute returns is probably not a good modeling choice.
3.  We assumed that volatility is changing slowly, or is ideally constant. This is not the case in practice. One approach is to impose a model on the time series of realized variances, so that we can produce out-of-sample forecasts. For example, a simple AR(1) model
    $$ \sigma_{t+1}^2 = \alpha + \beta \sigma_t^2 + \epsilon_{t+1} $$
    where $\epsilon_{t+1} \sim N(0,1)$.
4.  We ignored the distinction between open-to-close and close-to-open intervals. Close-to-open returns are often fundamentally driven. Also, we are ignoring the large volatility and bid-ask spreads in the first minutes of the trading day.

For the interested reader, McAleer and Medeiros (2008) and Andersen et al. (2013) compare a broad set of estimators, with several choices of parameters, for assets in different asset classes (equities, futures, indices). They use Romano and Wolf’s procedure for multiple comparison (Romano and Wolf, 2005) and the “model confidence set” (Hansen et al., 2011). They conclude that the RV estimator performs competitively across various assets and asset classes.[13]

---

**2.3 State-Space Estimation of Variance**

**2.3.1 MA(q) Original Model EWMA**

A very popular estimator of the expected value of a time series $\{x_t\}$ based on data up to time $t-1$ is the exponentially weighted moving average (EWMA). It takes the form
$$ \hat{x}_t = (1-K) \sum_{j=0}^\infty K^j x_{t-j-1} $$
for some $K \in (0,1)$. We discount the past by giving its observations exponentially decreasing weights, which makes sense, and even more so when we write the estimate as a recursion:
$$ \hat{x}_t = (1-K) x_{t-1} + K \hat{x}_{t-1} $$
A low value of $K$ forgets the past faster. The formula is computationally efficient both in terms of storage and of computation. For the uncentered variance estimation of a return, this takes the form:
(2.22)
$$ \hat{\sigma}_t^2 = (1-K) r_{t-1}^2 + K \hat{\sigma}_{t-1}^2 $$
**Insight 2.2: GARCH vs EWMA with an offset**
Recall Equation (2.11):
$$ h_t^2 = \frac{\alpha_0}{1-\beta_1} + \alpha_1 \sum_{j=1}^\infty \beta_1^{j-1} (r_{t-j}^2 - \frac{\alpha_0}{1-\beta_1}) $$
This is, save for an offset, very similar to Equation (2.22):
$$ \hat{\sigma}_t^2 = (1-K) \sum_{j=0}^\infty K^j r_{t-j-1}^2 $$
(we have changed the indexing convention to make it consistent with GARCH). The two are identical when $\alpha_0 = 0$, $\alpha_1 = 1-K$, and $\beta_1 = K$. Why not use GARCH then, provided that estimating the parameters per asset is doable? There is no fundamental objection. Commercial models favor simplicity (the half-life in each model, respectively $1/\log(1/K)$ or $1/\log(1/\beta_1)$ is often set to 20-60 days). If $K$ is chosen appropriately, the value of $\hat{x}_t$ is close to $h_t^2$, or because the performance improvement is not high enough to justify the additional complexity in parameter estimation and, in the case of commercial models, communication of the model’s characteristics to clients.
The half-life $T_{1/2}$ is such that $K^{T_{1/2}} = 1/2$, i.e., $T_{1/2} = \log(1/2) / \log K$.

In academic journals, GARCH receives relatively low attention compared to CARCH models. For example, RiskMetrics (J.P. Morgan, 1996) uses EWMA for forecasting volatility, including major commercial risk systems like RiskMetrics, Barra, and Axioma. It is the other way around. Aside from these practical considerations, is it possible to motivate the approach based on a model? We devote this section to understanding and extending this simple formula.

We will employ linear state-space models and Kalman filters, which are briefly covered in the Appendix, Section 2.4. Rather than giving a full general treatment, we will focus on a specific example. It is a simple model, but it is rich enough to illustrate the main ideas. As it happens, this example is also the simplest non-trivial example of a state-space model. The model (Muth, 1960) posits that there is a scalar state $x_t$ that evolves randomly over time with the addition of a Gaussian disturbance to its previous value. We observe the state imperfectly; the observation $y_t$ is a noisy measurement of $x_t$. In formulas:
$$ x_{t+1} = x_t + \tau_w \epsilon_{t+1} $$
$$ y_{t+1} = x_{t+1} + \tau_v \eta_{t+1} $$
$$ \epsilon_t \sim N(0,1), \quad \eta_t \sim N(0,1) $$
The innovations and the measurement noises are Gaussian with mean zero, and they are independent of each other (i.e., $\epsilon_t \perp \eta_s$ for all $t$ and $s$, and $\epsilon_t \perp \epsilon_s$ for all $t \ne s$, and $\eta_t \perp \eta_s$ for all $t \ne s$). I skipped the derivation; the interested reader can find it in the Appendix. Define the ratio of measurement to innovation noise as $\kappa := \tau_v^2 / \tau_w^2$. The stationary MLE estimate of the variance is given by
$$ \sigma_{t+1|t}^2 = \tau_w^2 \frac{1 + \sqrt{1 + (2\kappa)^2}}{2\kappa} $$
$$ \sigma_{t|t}^2 = \tau_w^2 \frac{1}{2\kappa} $$
and the optimal estimation recursion is
$$ K_t := \frac{\sigma_{t|t-1}^2}{\sigma_{t|t-1}^2 + \tau_v^2} $$
$$ \hat{x}_{t+1|t} = (1-K_t) \hat{x}_{t|t-1} + K_t y_t $$
The relationship between the measurement/innovation noise and $K_t$ is illustrated in Figure 2.3 for $y_t$ formula simplifies.

[Image: Figure 2.3 Relationship between $K$ and $\kappa := \tau_v^2 / \tau_w^2$. Plot shows K decreasing from 1 to 0 as kappa increases from 0 to 10.]

This is an exponential weighted average with a simple interpretation. Imagine that the state does not change at all. Then we want to use all of the history we can, since old observations and new ones are drawn from the same distribution. The half-life of EWMA is indeed long. Conversely, when the state changes at a rapid pace, i.e., $\kappa \ll 1$, then we want to discount the past very aggressively.

According to Muth’s original model applied to volatility estimation, the state is the instantaneous variance, and the observation $y_t$ is $r_t^2$, which is equal to $\sigma_t^2$ in expectation.

The model has obvious shortcomings. If returns are normally distributed, then the observation error is not normally distributed. More importantly, the model allows for negative values of the variance, and does not realistically model the evolution of the sum of iid innovations. Over time, the distribution of the variance becomes increasingly spread out: the standard deviation of the distribution grows as the square root of the number of periods. In practice, however, volatility appears to revert to its long-term average.

We cannot directly address the first problem. However, Kalman filters can work well with non-normal innovations and measurement errors, provided that these are not too heavy-tailed. As for the other shortcomings, we can refine the model to accommodate them. For example, we can introduce a mean-reverting model of variance, so that it behaves like an autoregressive process. We slightly extend the state equation by adding a mean-reversion term:
$$ x_{t+1} = \lambda x_t + (1-\lambda)\mu + \tau_w \epsilon_{t+1}, \quad \lambda > 0 $$
The state reverts to value $\mu$ when it is away from this equilibrium value. The stationary distribution of $x_t$ is Gaussian, with the expected value equal to $\mu$ and standard deviation equal to $\tau_w / \sqrt{1-\lambda^2}$. The optimal variance estimator is still
$$ \hat{x}_{t+1|t} = (1-K_t) \hat{x}_{t|t-1} + K_t y_t $$
However, compared to the first model, the value of $K_t$ is smaller. Otherwise stated, the mean reversion term makes the distribution of the true variance more concentrated around its long-term mean. This implies that we discount the past less. The detailed derivation of these formulas is in the Appendix, Section 2.4.2.

**2.3.2 The Harvey-Shepherd Model**

As a final example of the flexibility that linear state-space models can offer, I present the model by Harvey and Shephard (1996), which has several desirable features: it has a closed-form solution, the volatility is by design positive and the distribution of the volatility itself is log-normal, hence right-skewed, as we would expect, and the stock returns are locally log-normal.
The generating process for returns $r_t$ is assumed to be
(2.23)
$$ r_t = \exp(x_t/2) \xi_t - 1 $$
where $\beta$ is a known constant, and $\xi_t \sim N(0,1)$. Hence, returns are, at any point in time, log-normally distributed. Define
$$ u_t := \log(1+r_t) - \beta $$
$$ \Rightarrow \quad \log u_t^2 = x_t + \log \xi_t^2 $$
where
$$ \gamma := E(\log \xi_t^2) \approx -1.27 $$
, and $\xi_t$ is a zero-mean random variable with standard deviation given by
$$ \text{stdev}(\log \xi_t^2) \approx 2.22 $$
Define
$$ y_t := \log u_t^2 - \gamma $$
$$ y_t = [\log(1+r_t) - \beta]^2 - \gamma $$
so that we get an observation equation:
$$ y_t = x_t + \eta_t $$
Now, we posit an evolution equation for $x_t$:
$$ x_{t+1} = b + a x_t + \epsilon_t $$
This is the same model as (2.8), from which we obtain an estimate $\hat{x}_{t+1|t}$. If $\beta=0$, then the formulas take a simple form for $y_t = \log(1+r_t)$, and the state estimate is given by
$$ \hat{x}_{t+1|t} = (1-K) \hat{x}_{t|t-1} + K [\log(1+r_t)] - \gamma $$
Since
$$ r_t = \exp(\hat{x}_{t|t-1}/2) \xi_t - 1 $$
is a log-normal random variable, the estimated standard deviation of $r_t$ is
$$ \hat{\sigma}_{t+1|t} = \sqrt{(\exp(\hat{\sigma}_{\hat{x}_{t+1|t}}^2) - 1) \exp(2\hat{x}_{t+1|t})} $$
A simplified Harvey-Shepherd model starts with Equation (2.23), in which it assumes the first-order approximation $\gamma^2-1 \approx \gamma$, and the parameter $\beta=0$
$$ r_t = \exp(x_t/2) \xi_t $$
Define
$$ \log r_t^2 = x_t + \log \xi_t^2 $$
$$ = x_t + \eta_t + \gamma $$
where $\eta_t$ and $\xi_t$ are defined as for the Harvey-Shepherd model above. The model is completed by the following equations, also from the original model:
$$ x_{t+1} = b + a x_t + \epsilon_t $$
$$ y_t = \log r_t^2 - \gamma $$
The state and volatility estimates are
$$ \hat{x}_{t+1|t} = (1-K) \hat{x}_{t|t-1} + K [\log r_t^2 - \gamma] $$
$$ \hat{\sigma}_{t+1|t}^2 = e^{\hat{x}_{t+1|t}/2} $$

---

**2.4 Appendix**

**2.4.1 The Kalman Filter**

This section contains a short treatment of the Kalman filter (KF). The KF is in its modern form dates to the early 1960s, with work by Kalman (1960a,b). Its intellectual lineage can be traced back to Gauss’s work on least squares. It has become available that made calculations feasible in real time. This made the rediscovery of the filter by Kalman very timely. Rockets used by the Apollo program contained implementations of the KF as early as 1968. Since then, the study of linear control and filtering has blossomed. There are several excellent books on the KF, and there are several monographs covering the KF in detail from different viewpoints: econometric (Whittle, 1996; Simon, 2006), statistical (Harvey, 1993), and econometric (Hansen and Sargent, 2013). I cover the KF because, for somewhat mysterious reasons, the derivation of the KF is often more complicated than it should be. A rigorous, yet accessible proof essentially fits in half a page and should save the reader a few hours.

We need the following elementary fact. Let $\mathbf{z}' = [\mathbf{x}', \mathbf{y}']'$ be a multivariate normal random vector with mean and covariance matrix
$$ \mu_{\mathbf{z}} = \begin{bmatrix} \mu_{\mathbf{x}} \\ \mu_{\mathbf{y}} \end{bmatrix}, \quad \text{COV}(\mathbf{z}) = \Sigma_{\mathbf{z}} = \begin{bmatrix} \Sigma_{\mathbf{x}\mathbf{x}} & \Sigma_{\mathbf{x}\mathbf{y}} \\ \Sigma_{\mathbf{y}\mathbf{x}} & \Sigma_{\mathbf{y}\mathbf{y}} \end{bmatrix} $$
The random vector $\mathbf{x}$, conditional on $\mathbf{y}=\mathbf{b}$, is still normally distributed, with conditional mean and covariance matrix equal to
(2.24)
$$ E(\mathbf{x}|\mathbf{y}=\mathbf{b}) = \mu_{\mathbf{x}} + \Sigma_{\mathbf{x}\mathbf{y}} \Sigma_{\mathbf{y}\mathbf{y}}^{-1} (\mathbf{b} - \mu_{\mathbf{y}}) $$
(2.25)
$$ \text{COV}(\mathbf{x}|\mathbf{y}=\mathbf{b}) = \Sigma_{\mathbf{x}\mathbf{x}} - \Sigma_{\mathbf{x}\mathbf{y}} \Sigma_{\mathbf{y}\mathbf{y}}^{-1} \Sigma_{\mathbf{y}\mathbf{x}} $$
This can be verified directly by integration.

Our model has two components. The first is a state, represented by a random vector $\mathbf{x}_t$. This vector follows a simple evolution rule: $\mathbf{x}_{t+1} = A \mathbf{x}_t + \epsilon_{t+1}$. The vector $\epsilon_t$ is random, serially independent (i.e., $\epsilon_t \perp \epsilon_s$ for $t \ne s$), and distributed according to a multivariate normal distribution. The state $\mathbf{x}_t$ is not observable directly; the only thing we know is its probability distribution at time $t$. We assume its mean is $\hat{\mathbf{x}}_{t|t-1}$ and its covariance matrix is $\Sigma_{t|t-1}$. The second component is a vector $\mathbf{y}_t$, which is a linear transformation of $\mathbf{x}_t$, corrupted by noise:
$$ \mathbf{y}_{t+1} = B \mathbf{x}_{t+1} + \eta_{t+1} $$
Once you read Chapter 5, you will note the similarity with the factor model equation:
$$ \text{state} \leftrightarrow \text{factor return} $$
$$ \text{observation} \leftrightarrow \text{asset return} $$
The vector $\eta_t$ is random, serially independent, independent of $\epsilon_t$ ($ \eta_t \perp \epsilon_s, \dots$), and distributed according to a multivariate normal distribution.
Summing up, the distributions of $\mathbf{x}_t, \epsilon_t, \eta_t$ are given by
$$ \mathbf{x}_t \sim N(\hat{\mathbf{x}}_{t|t-1}, \Sigma_{t|t-1}) $$
$$ \epsilon_t \sim N(0, \Sigma_\epsilon), \quad \eta_t \sim N(0, \Sigma_\eta), \quad t \ne s $$
and the linear state-space model is given by
(2.26)
$$ \mathbf{x}_{t+1} = A \mathbf{x}_t + \epsilon_{t+1} $$
(2.27)
$$ \mathbf{y}_{t+1} = B \mathbf{x}_{t+1} + \eta_{t+1} $$
I denote $\hat{\mathbf{x}}_{t+1|t}$ the conditional estimates for the mean and covariance matrix of the state $\mathbf{x}_{t+1}$, based on information $\mathcal{F}_t = \{\mathbf{y}_1, \dots, \mathbf{y}_t\}$. And I denote $\hat{\mathbf{x}}_{t+1|t+1}$, the estimates based on information $\mathcal{F}_{t+1}$.
The vector $\mathbf{z}_t$ is defined as the combination of state and observation:
$$ \mathbf{z}_t := \begin{bmatrix} \mathbf{x}_t \\ \mathbf{y}_t \end{bmatrix} $$
Based on information up to time $t-1$, the covariance of $\mathbf{z}_t$ is
$$ \text{Cov}(\mathbf{z}_t) = \begin{bmatrix} \Sigma_{t|t-1} & \Sigma_{t|t-1} B' \\ B \Sigma_{t|t-1} & B \Sigma_{t|t-1} B' + \Sigma_\eta \end{bmatrix} $$
We observe $\mathbf{y}_t$. The vector $\mathbf{x}_t$ is normally distributed. We compute the conditional covariance of $\mathbf{x}_t$ given $\mathbf{y}_t$ using Equations (2.24) and (2.25).
$$ \Sigma_{t|t} = \Sigma_{t|t-1} - \Sigma_{t|t-1} B' (B \Sigma_{t|t-1} B' + \Sigma_\eta)^{-1} B \Sigma_{t|t-1} \quad \text{(update step)} $$
$$ = \Sigma_{t|t-1} (I - B' (B \Sigma_{t|t-1} B' + \Sigma_\eta)^{-1} B \Sigma_{t|t-1}) $$
$$ \hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \Sigma_{t|t-1} B' (B \Sigma_{t|t-1} B' + \Sigma_\eta)^{-1} (\mathbf{y}_t - B \hat{\mathbf{x}}_{t|t-1}) $$
Once we have the posterior distribution given the observation $\mathbf{y}_t$, the conditional distribution of $\mathbf{x}_{t+1}$ follows from Equation (2.26) is Gaussian with the following conditional mean and covariance matrix:
(2.28)
$$ \Sigma_{t+1|t} = A \Sigma_{t|t} A' + \Sigma_\epsilon \quad \text{(prediction step)} $$
(2.29)
$$ \hat{\mathbf{x}}_{t+1|t} = A \hat{\mathbf{x}}_{t|t} + A \Sigma_{t|t-1} B' (B \Sigma_{t|t-1} B' + \Sigma_\eta)^{-1} (\mathbf{y}_t - B \hat{\mathbf{x}}_{t|t-1}) $$
The measurement and time update equations above are the whole of the KF. If we combine Equation (2.28) and (2.29), the covariance matrix evolves according to the equation
$$ \Sigma_{t+1|t} = A \Sigma_{t|t-1} A' - A \Sigma_{t|t-1} B' (B \Sigma_{t|t-1} B' + \Sigma_\eta)^{-1} B \Sigma_{t|t-1} A' + \Sigma_\epsilon $$
This is called a Riccati recursion. In steady state the covariance matrix does not change in consecutive periods: $\Sigma_{t+1|t} = \Sigma_{t|t} = \Sigma$. We can solve for the stationary matrix
$$ \Sigma = A \Sigma A' - A \Sigma B' (B \Sigma B' + \Sigma_\eta)^{-1} B \Sigma A' + \Sigma_\epsilon $$
This is a discrete-time algebraic Riccati equation.
The matrix
$$ K_t = \Sigma_{t|t-1} B' (B \Sigma_{t|t-1} B' + \Sigma_\eta)^{-1} $$
is called the optimal Kalman gain. The equations become
(2.30)
$$ \Sigma_{t|t} = (I - K_t B) \Sigma_{t|t-1} $$
(2.31)
$$ \hat{\mathbf{x}}_{t|t} = (I - K_t B) \hat{\mathbf{x}}_{t|t-1} + K_t \mathbf{y}_t $$
(2.32)
$$ \hat{\mathbf{x}}_{t+1|t} = A \hat{\mathbf{x}}_{t|t} = A \Sigma_{t|t} A' + \Sigma_\epsilon $$
(p. 93) $\hat{\mathbf{x}}_{t+1|t} = A \hat{\mathbf{x}}_{t|t}$

**2.4.2 Kalman Filter Examples**

Example 2.1:
(Muth, 1960)
$$ x_{t+1} = x_t + \tau_w \epsilon_{t+1} $$
$$ y_{t+1} = x_{t+1} + \tau_v \eta_{t+1} $$
This is the simplest possible state-space model. The stationary estimate $\hat{\sigma}_{t+1|t}^2$ is given by the solution to the Riccati equation:
$$ \hat{\sigma}_{t+1|t}^2 = \tau_w^2 \frac{1+\sqrt{1+(2\kappa)^2}}{2\kappa} $$
$$ \hat{\sigma}_{t|t}^2 = \tau_w^2 \frac{1}{2\kappa} $$
$$ K = \frac{\hat{\sigma}_{t|t-1}^2}{\hat{\sigma}_{t|t-1}^2 + \tau_v^2} $$
$$ \hat{x}_{t+1|t} = (1-K) \hat{x}_{t|t-1} + K y_t $$
where we have reintroduced the parameter
$$ \kappa = \tau_v^2 / \tau_w^2 $$
Loosely, this is a noise-to-signal ratio. $K$ is high when the measurement error is high compared to the optimal change of the state per period. For $\kappa \gg 1$, the formula simplifies to $K \approx 1/(1+\kappa)$.
$$ \hat{x}_{t|t} = \frac{\kappa}{1+\kappa} \hat{x}_{t|t-1} + \frac{1}{1+\kappa} y_t $$
Example 2.2:
(AR(1) model) In this model, the state equation is
$$ x_{t+1} = \alpha x_t + \tau_w \epsilon_{t+1} $$
If there is a mean-reverting process, introduce a long-term mean value $\mu > 0$ and a situation contains $\lambda = \beta_1$, and set
$$ (y_t|\mathcal{F}_{t-1}) \sim N(\mu, \lambda) $$
Equation (2.26) becomes
$$ x_{t+1} = \lambda x_t + (1-\lambda)\mu + \tau_w \epsilon_{t+1} $$
The state reverts to value $\mu$ when it is away from this equilibrium value. The stationary distribution of $x_t$ is Gaussian, with mean $\bar{\mu}$ and standard deviation $\tau_w / \sqrt{1-\lambda^2}$.
Define
(2.33) $u_t := x_t - \bar{\mu}$
(2.34) $v_t := y_t - \bar{\mu}$
We rewrite the equation as
$$ u_{t+1} = \lambda u_t + (1-\lambda)(\mu - \bar{\mu}) + \tau_w \epsilon_{t+1} $$
$$ v_{t+1} = u_{t+1} + (1-\lambda)(\mu - \bar{\mu}) + \tau_v \eta_{t+1} $$
The state-space equations are
$$ u_{t+1} = a u_t + \tau_w \epsilon_{t+1} $$
$$ v_{t+1} = u_{t+1} + \tau_v \eta_{t+1} $$
The Riccati equation is
$$ (1-\kappa^2) \hat{\sigma}_{t+1|t}^4 - \tau_w^2 \hat{\sigma}_{t+1|t}^2 - \tau_w^4 = 0 $$
$$ \Rightarrow \quad \hat{\sigma}_{t+1|t}^2 = \frac{\tau_w^2}{2(1-\kappa^2)} [1 + \sqrt{1+4(1-\kappa^2)}] $$
$$ = \frac{\tau_w^2}{2(1-\kappa^2)} [1 + \sqrt{1+4\kappa^2-4\kappa^4}] $$
$$ = \frac{\tau_w^2}{2(1-\kappa^2)} [1 + \sqrt{(1-2\kappa^2)^2 + 4\kappa^2(1-\kappa^2)}] $$
$$ = \frac{\tau_w^2}{2(1-\kappa^2)} [1 + \sqrt{(1-2\kappa^2)^2}] $$
$$ = \frac{\tau_w^2}{2(1-\kappa^2)} [1 + |1-2\kappa^2|] $$
$$ = \begin{cases} \frac{\tau_w^2}{1-\kappa^2} & \text{if } |1-2\kappa^2| = 1-2\kappa^2 \\ \frac{\tau_w^2 \kappa^2}{1-\kappa^2} & \text{if } |1-2\kappa^2| = -(1-2\kappa^2) \end{cases} $$
$$ = \tau_w^2 \frac{1}{1-\kappa^2} \left(1 - \frac{2\kappa}{1+\sqrt{1+(2\kappa)^2}}\right) $$
$$ K = \frac{\hat{\sigma}_{t|t-1}^2}{\hat{\sigma}_{t|t-1}^2 + \tau_v^2} $$
Now replace $K_t$ using Equation (2.30) and (2.32).
$$ \hat{x}_{t+1|t} = (1-K) \hat{x}_{t|t-1} + K y_t $$
For $\kappa \ll 1$, the formula is identical to that of Example 2.1. It is straightforward to verify that $K_t$ is decreasing in $\kappa$, and consequently also $K$ is decreasing in $\alpha$. There are two insights to be drawn from this:
1.  The EWMA is still an optimal estimator for a mean-reverting model of volatility.
2.  In the process of mean reversion, $K$ decreases, everything else being equal. We discount the past less, because mean reversion causes future conditional volatility to be more concentrated than past conditional volatility is emerging less from past data, and observations become more informative.

---

**2.5 Exercises**

**Exercise 2.1:**
Prove that
$$ \left[ \prod_{t=1}^T (1+r_t) \right]^{1/T} - 1 \le \frac{1}{T} \sum_{t=1}^T r_t $$

**Exercise 2.2:**
Provide an example of two random variables that are uncorrelated but dependent.

**Exercise 2.3:**
Provide a second example, employing an entirely different rationale for the lack of correlation from the first one.

**Exercise 2.4:**
With respect to Equation (2.10), prove that if $E[h_0^2]$ is finite, i.e., $E[h_0^2] < \infty$, a stationary distribution exists, i.e.,
$$ E_t[\log(\beta_1 + \alpha_1 \epsilon_t^2)] < 0 $$
(Hint: Use Jensen’s inequality.)

**The Takeaways**

1.  Importance of Univariate Models. Understanding single-asset returns is foundational for modeling portfolio returns, especially in terms of volatility. GARCH and exponential moving averages are key tools for this.
2.  Types of Returns. Definitions include simple returns, dividend-adjusted returns, compounded returns, and log returns. Excess returns, adjusted for a risk-free rate, are also essential in portfolio management.
3.  Stylized Facts of Returns. Common features of returns include a lack of autocorrelation, heavy tails, volatility clustering, and aggregational Gaussianity.
4.  GARCH Models. GARCH(1,1) captures some stylized facts like heavy tails and volatility clustering, but has limitations, such as overly high long-term memory.
5.  Realized Volatility. Estimating volatility directly using high-frequency data (e.g., tick or minute-by-minute) is effective, though market microstructure effects (e.g., bid-ask spreads) and thinly traded securities pose challenges.
6.  EWMA as a Volatility Estimator. Exponentially weighted moving averages offer a practical, interpretable method for estimating volatility, often used in commercial risk systems.
7.  State-Space Models. Kalman filters and state-space models provide a structured way to estimate time-varying volatility, allowing for enhancements like mean reversion in volatility.

**Notes** (Right Sidebar from Page 99)

1.  This word comes from the Latin *summaarius*, or “a number”.
2.  Defined in Section 1.2.
3.  Geometric returns and dividend-adjusted returns are in Connor et al. (2010) and Ruppert and Matteson (2015).
4.  The two rates are not exactly the same: when borrowing, the effective rate charged to the borrower by the lending institution is the risk-free rate plus a spread; when lending (holding cash), the rate paid to the investor is the risk-free rate minus a spread. For modeling purposes, we consider them identical.
5.  As of publication time, the minimum tick size is $0.01 in U.S. exchanges for stocks trading above $1.
6.  A detailed discussion of the Roll model and its extensions is in Hasbrouck (2007).
7.  Note, however, that I am not including the leverage effect among the stylized facts. In the words of Cont (2001), “Most measures of volatility of an asset are negatively correlated with the returns of that asset.” This effect is not sufficiently robust across asset data, as shown by Ruffini-Crouzet et al. (2019). Whether to take it into account or not is left to the reader, and on the specific application they are considering.
8.  John Cochrane has written extensively on the subject, e.g., Cochrane (2005) and the long story “Predictability and Covariation” (Cochrane, 2011).

---
(Notes continued from Page 103)

9.  The literature on GARCH models alone is immense. Zivot and Wang (2003); Ghalanos (2019); Tsay (2010); Clark et al. (2006); Ruppert and Matteson (2015); Bollerslev (2008) are good starting points. The original ARCH model is in Engle (1982). The GARCH model is in Bollerslev (1986). The book Andersen et al. (2009) has dedicated chapters covering univariate GARCH, EGARCH (Teräsvirta, 2009b), moments of GARCH models (Teräsvirta, 2009a), their statistical inference properties (Davis and Mikosch, 2009), and multivariate GARCH. For a review of the empirical performance of GARCH, TARCH, GJR-GARCH, and a few other models, see Hansen and Lunde (2005); Brownlees et al. (2011).
10. The convergence properties of Random Recursive Equations (RREs) were studied by Kesten (1973). Diaconis and Freedman (1999) further developed the theory. The general form of an RRE is $X_{n+1} = f(X_n, Y_{n+1})$, where $\{Y_n\}_{n \ge 1}$ is an iid random sequence, of which RREs are a special case. A monograph on RREs, covering both the univariate and multivariate case, is Buraczewski et al. (2016).
11. The kurtosis of log returns, which is to a first approximation close to the daily Sharpe Ratio computed on returns.
12. An early analysis of the “vanilla” realized variance estimator is Barndorff-Nielsen and Shephard (2002), and a survey is Andersen and Benzoni (2009). Also relevant are the surveys of Andersen et al. (2001b, 2003), which introduced realized volatility as a measure of risk management purposes. Essential readings on realized volatility estimators are Zhang et al. (2005), which presents several estimators and introduces the idea of sub-sampling for RV; the series of papers Barndorff-Nielsen et al. (2008, 2009, 2011), which introduced several estimators, the two-scales RV estimator (Zhang et al., 2005), the multi-scale RV estimator (Aït-Sahalia et al., 2011), and the kernel RV estimator. This list of estimators is not exhaustive. For example, Hansen and Lunde (2006a) analyze an autocorrelation-adjusted estimator introduced by French et al. (1987). Different estimators are studied by Patton (2011) and Liu et al. (2015), and the minimum bias estimator is by Bandi and Russell (2008). However, these estimators depend on several parameters, like sampling and subsampling intervals, or the choice of kernel.
13. There are a few cases where this is not true. When high-frequency measurements are available, the estimator is contaminated by a microstructure noise term $O_p(1/\sqrt{n})$ for fixed sampling intervals (Barndorff-Nielsen et al., 2008b). In addition, at lower frequencies, a systematic truncated RV (Mancini, 2001, 2009) also outperforms vanilla RV; see Liu et al. (2015).
14. In the notation of the book (e.g., Eq. (2.24)), we use the following properties for random vectors $\mathbf{z}, \xi$ and a commensurable matrix $\mathbf{B}$:
    $$ \text{cov}(B\xi, \mathbf{z}) = B \text{cov}(\xi, \mathbf{z}) $$
    and
    $$ \text{cov}(\xi, B\mathbf{z}) = \text{cov}(\xi, \mathbf{z}) B' $$

```

Okay, here is the Markdown compilation for Chapter 3.

```markdown
Everand

