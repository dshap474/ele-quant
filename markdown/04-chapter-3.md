**Chapter 3**
**Interlude: What Is Performance?**

**The Questions**

1.  What are the key performance metrics for a portfolio manager?
2.  How are they related to each other?

A discretionary portfolio manager satisfies; a quantitative portfolio manager optimizes. While such a statement is overly flat, a discretionary portfolio manager, like Odysseus, may optimize too, like using shark repellent for food intake or migrating warblers minimize traveling distance to Cuba. On the other side, optimization is part of the job description of a quant. This chapter introduces and justifies the investment metrics that will appear in later chapters. In some cases, the metrics enter the objective function directly or indirectly, through constraints. The role played by objectives and constraints is, to some extent, interchangeable.

The performance metrics of a portfolio manager are, by and large, these:

*   The expected return of the strategy.
*   The volatility of the strategy.
*   The Sharpe ratio and the Information ratio.
*   Capacity.

---

**3.1 Expected Return**

The expected return of the strategy, $\mu$, defined as the rate of Profit and Loss (P&L) to Assets Under Management (AUM). With the possible exception of Mother Teresa, investors prefer more money to less, and returns are an adequate way to describe this. Returns are preferred to actual money because the normalization makes the measure stationary, i.e., having (approximately) the same distribution across different time periods. The interview (or vignette) for estimating $\mu$ is independent of the universe of assets. The same considerations apply to comparison across assets and across funds. Returns can be optimized either over the course of an investment period or over the lifetime of the strategy. In practice, the two problems are separable: we solve a sequence of single-period problems, which we embed in a larger multi-period problem.

---

**3.2 Volatility**

We introduced the volatility of returns in Section 2.1. The use of volatility is vexing. It is ubiquitous, which, however, does not make it justified. There are, however, several arguments in favor of volatility. The first one is empirical. Cont (2001) reports a tail index for stock returns that is greater than 2 but smaller than 5; only fourth moments of returns may be finite. Therefore volatilities are finite and can be estimated. But there is no assurance that skewness or kurtosis may be estimated. The second reason is related to specific assumptions, described below.

Say that the investor, whose single-period utility maximization problem of the form $\max_w E[u(w'R)]$, the utility function describes the preferences of the investor, and is increasing and concave. Then, one can justify the use of volatility using three different sets of assumptions (Huang and Litzenberger, 1988):

1.  The utility function is well approximated by a second-order Taylor expansion centered at the expected payoff, so that
    $$ u \approx a E(w'R) - b E[(w'R - E(w'R))^2] $$
    ...and the returns are arbitrarily distributed but with finite variance.
2.  The utility takes a specific form (aside from the above quadratic utility):
    $$ u(w'R) = \begin{cases} - \exp(-w'R) \\ (w'R)^{1-\gamma} & \gamma \in (0,1) \\ \log(w'R) \end{cases} $$
3.  Lastly, one can assume arbitrary utility and returns to be normally distributed.

As we discussed in Chapter 2, there is agreement that returns of many securities have finite variance. The assumption that we can approximate utility with a quadratic function is not unrealistic, since the optimization horizon or quantitative investing is short, and the payoffs are small. Quadratic utility is not quadratic, and this will be exploited by the active asset manager. The risk aversion parameter, often $b$. A higher value of the ratio $b/a$ can be interpreted as penalizing more the uncertainty of payoffs relative to their expected values, i.e., being more risk-averse.[1] I am not discussing this further, since the topic is covered extensively in textbooks, such as Huang and Litzenberger (1988), and is not essential to the remainder of this book.

---

**3.3 Sharpe Ratio**

The Sharpe Ratio (SR) is defined as the ratio of expected excess returns of a strategy to its volatility over the previous two quantities by measuring returns in units of volatility over a certain period of time. The Information Ratio (IR) is defined analogously to the Sharpe Ratio but employs a different type of returns: the idiosyncratic returns of a strategy. These returns will be introduced in Chapter 5; for now, it suffices to say that active asset management returns describe the “intrinsic” returns of the strategy, i.e., the component of returns that is not driven by common factors affecting all returns at once.

If we assume that the returns $\xi$ of a strategy are identically distributed and independent, the Sharpe Ratio is the same as the t-statistic of the mean of the return distribution. In finance, the Sharpe Ratio is named after William F. Sharpe, one of the authors of the Capital Asset Pricing Model (CAPM). The Sharpe Ratio has drawbacks. For example, it is not additive across assets, it is not well defined as a metric that ranks uncertain outcomes. Aside from decision-theoretic considerations,[2] the Sharpe Ratio of a portfolio with negative expected return of $-\sigma$ and volatility $2\sigma$ is lower than the Sharpe Ratio with the same negative return, and volatility $0.5\sigma$. This is counter-intuitive at best and wrong at worst. Second, it assumes that investors’ risk attitude is measured solely by volatility. This implicitly assumes that there is only one source of risk, of which there is a near-infinite supply.[3] There are advantages, however. First, the Sharpe Ratio is intuitive, return in units of “risk.” Second, it comes with a rich arsenal of theoretical results. We have confidence intervals and generalizations of the empirical properties of portfolio returns. The Sharpe Ratio is related to the probability of underperforming the risk-free rate. The Sharpe Ratio also implies a bound on the probability of incurring a certain loss. This follows from Cantelli’s inequality. For a random variable $\xi$ with mean $\bar{\mu}$ and standard deviation $\bar{\sigma}$, this inequality states that
$$ P(\xi < \bar{\mu} - \lambda) \le \frac{\bar{\sigma}^2}{\bar{\sigma}^2 + \lambda^2} $$
$$ \Rightarrow \quad P(\xi < -\lambda) \le \frac{\bar{\sigma}^2}{\bar{\sigma}^2 + (\lambda + \bar{\mu})^2} $$
If $\xi$ is the annual return of a strategy, and SR is the annualized Sharpe Ratio of the strategy, and the loss is expressed as a multiple of standard deviations $-L\sigma$, an practitioner often asks, then the inequality is
$$ P(\xi < -L\sigma) \le \frac{1}{1 + (L + \text{SR})^2} $$

(Right Sidebar Text from Page 110)

This holds for any distribution of returns with Sharpe Ratio SR. For example, consider an annualized Sharpe Ratio of 1 and an annualized volatility of $L=3\sigma$. The probability of a loss of more than three standard deviations is not greater than $1/(1+(3+1)^2) = 1/17 \approx 6\%$. This is a much looser bound than one would obtain under the assumption of normal returns. In that case, the probability of a loss would be a $P(Z < -3) \approx 0.1\%$.

**FAQ 3.1: What are the dimensions of the Sharpe Ratio?**
Return, volatility, and Sharpe Ratio depend on the time horizon over which they are measured. Daily return vs. daily PnL, etc. Annualized return, where we assume the mean daily PnL (return) is $\bar{\mu}$ and volatility is $\bar{\sigma}$.[4] We assume that returns have zero mean, so that $\bar{\mu} = 0$. An example: a strategy has a return of $x\% / \sqrt{N}$ days / $x\% / N$ days. Provided that returns are serially uncorrelated (see Section 2.1.5), the variance of the sum of $T$ identically distributed returns is equal to $T$ times the variance of the single-period returns. The square root of variance and has the dimension $[\text{time}]^{-1/2}$. The Sharpe Ratio has the dimension $[\text{time}]^{-1/2}$.
$$ [\text{time}]^{-1/2} / [\text{time}]^{-1/2} = [\text{time}]^{-1/2} $$
When converting the horizon of a Sharpe Ratio for an equity strategy from a daily horizon to a monthly one, we multiply the daily Sharpe Ratio by $\sqrt{T_M}$, where $T_M$ is the number of trading days in a month (typically 21). The most common annualized Sharpe Ratio in the United States is $\sqrt{252}$.
[4] The number of trading days depends on the country, the asset class, and the year. In the United States, as of 2014, the stock market is open 252 days a time.

**FAQ 3.2: What is the confidence interval for the Sharpe Ratio?**
Suppose you observe $T$ consecutive returns (or PnLs), and estimate the Sharpe Ratio from these data. What is the confidence interval of this estimator? First, the Sharpe Ratio estimator is
$$ \hat{\mu} = \frac{1}{T} \sum_{t=1}^T r_t, \quad \hat{\sigma}^2 = \frac{1}{T} \sum_{t=1}^T (r_t - \hat{\mu})^2 $$

---
(Continuation from Page 114)

For excess returns $r_t$ that are iid and with finite variance, the estimator is normally distributed in the limit $T \to \infty$, with standard error
$$ \text{SR} = \frac{\hat{\mu}}{\hat{\sigma}} $$
$$ \text{SE(SR)} = \sqrt{\frac{1 + \text{SR}^2/2}{T}} $$
Compare this to the case in which we knew in advance the standard deviation of the returns. The Sharpe Ratio is then $\text{SR} = \hat{\mu}_0 / \sigma$ and the SE is $\sigma/\sqrt{T}$.
In the case of autocorrelated returns with $\text{corr}(r_t, r_{t-1}) = \rho \ne 0$, the Sharpe Ratio estimator is the cross-sectional $\hat{\mu}'$:
$$ \text{SR}_{\text{adj}} = \frac{\hat{\mu}}{\hat{\sigma}} \sqrt{\frac{1-\rho}{1+\rho}} \approx \frac{\hat{\mu}}{\hat{\sigma}} (1-\rho) $$

---

**3.4 Capacity**

Whereas the returns and the Sharpe Ratio are well known and defined, the capacity of a strategy is not unequivocally defined. Its informal definition is “capacity is the highest expected P&L that a strategy is able to produce over a certain horizon”. You may ask, “but isn’t expected P&L just equal to Sharpe Ratio times dollar volatility?” Capacity is essentially the maximum volatility at which we can run a strategy.[5] This would be fine if the Sharpe Ratio were independent of volatility, but in that case, capacity would be infinite. Sharpe Ratio is, at best, flat for low values of volatility; however, it almost always a decreasing function of volatility. For a large enough volatility, the Sharpe Ratio becomes zero, and beyond this threshold the strategy is unprofitable. The capacity of a strategy can be defined as the maximum P&L that can be achieved, subject to a constraint that the Sharpe Ratio exceeds a certain acceptable level. Alternatively, we could require a minimum bound on the expected return in capital. Defined this way, the capacity is an important parameter for hedge fund managers and portfolio managers alike. A strategy may have attractive return and Sharpe Ratio when run at low volatility. If it can yield only a modest P&L, it will be economically unattractive.

**The Takeaways**

1.  Expected Return: Represents profit relative to assets under management. It's stationary and intensive, allowing for comparisons across funds and periods.
2.  Volatility: A standard measure in portfolio optimization, particularly useful for assets with finite variance, used under assumptions about investor utility functions.
3.  Sharpe Ratio: Represents returns per unit of "risk" (volatility), widely adopted but has limitations such as dependence on risk aversion choice and unsuitability for outcomes with negative returns.
4.  Capacity: Refers to the maximum Profit and Loss (P&L) achievable within a strategy's acceptable risk (Sharpe Ratio) limits, crucial for assessing the economic viability of a strategy.

**Notes**

1.  [1] A sufficient condition for the asymptotic consistency of the variance estimator is that the fourth moment be finite.
2.  [2] Or, more commonly, the excess returns, i.e., the returns of a strategy in excess of the risk-free rate. We assume, for simplicity only, that the cost of holding a self-financing security: we borrow one dollar in the first period at the risk-free rate, and buy one dollar of the security. In the second period, we receive the security return, and pay off the loan.
3.  [3] See Huang and Litzenberger (1988).
4.  [4] For these, see [Skipping text] for a list of risk metrics, both theoretically justified and heuristic.

```

Okay, I will extract the text from the screenshots that belong to Chapter 4 and compile it into a single Markdown file.

```markdown
Everand

