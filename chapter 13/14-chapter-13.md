**Chapter 13**
**Dynamic Risk Allocation**

**The Questions**

1.  What are the limitations of single-period portfolio optimization for long-term investment strategies?
2.  How does the Kelly criterion provide both theoretical insights and practical guidance for capital allocation and investment decisions?
3.  In what ways do different capital allocation strategies impact cumulative returns and risk profiles?
4.  How do parameter uncertainty and fractional Kelly allocation strategies improve risk management for investors?
5.  What are the mathematical properties of Kelly strategies that make them desirable for long-term growth?
6.  How are stop-loss policies related to the Kelly criterion?

So far we have focused exclusively on single-period portfolio optimization. This may be appropriate for one-off investment decisions, but is inadequate for long-term investment strategies. This is a trite academic statement because an uninterrupted consequence of decisions is a trite modeling statement: that we make decisions in sequence. Investment in the long run, both at the level of the individual consumer and at the aggregate level. Much of the literature has been ignored by asset managers, for a few possible reasons we comment on below. First, these models require the specification of a utility function and of a discount factor (a discount factor for future utility), something that no investment manager would or could specify. If Quadrifoglio utility had been the main justification of MVO, it would probably have never been adopted. Second, these models don’t capture the institutional reality of asset managers. Consumption of asset managers responds to inflows, not just their own preferences. Asset managers’ utility, if any, is driven by inflows. Moreover, outflows do not bear a direct relationship to the principals’ utilities (i.e., those who provide the investment capital to the manager). The reason for this is that inflows and outflows occur at low rates (they are “sticky”), due to inertia of the principals who avoid changing their asset allocations and to search costs. Third, even with the asset managers (who require long allocations relative to capital withdrawal).

Even if hedge fund managers do not read the academic literature while driving their Ferraris (don’t read and drive), they still have to make decisions about the risk they want to take the next day, week, and month. There is one line of research, initiated by electrical engineers and mostly developed by researchers not employed by economics departments, which is relevant to investors. It is broadly known as the Kelly criterion, Kelly Formula, Kelly Gambling, Growth Portfolios, Optimal Growth Portfolios, or Optimal Growth Investing. It has both descriptive power, in that it is followed by many successful investors, and prescriptive value, in that it is based on first principles and has attractive properties. The rest of this chapter is devoted to presenting the basics of the theory. We start with a simple example and motivation for the Kelly criterion. Then we move to the properties of Kelly, its relation to portfolio optimization and Sharpe Ratio. The way that vodka is not suitable as a dinner drink, “Pure” Kelly is not suitable for investing. Fractional Kelly is to Kelly what Chardonnay is to vodka: more sustainable, better tasting and ultimately more fun. Finally, we introduce a time-varying version of fractional Kelly, which helps manage the occurrence of drawdowns.

---

**13.1 The Kelly Criterion**

To introduce the ideas behind the Kelly criterion, we consider first a very simple example. We have one risky asset in which to invest, which returns `r` with probability `p` or `-r` with probability `1-p`. The single-period expected return of the asset is `μ = (2p-1)r`, and its volatility is `σ = 2r√(p(1-p))`. You have to decide how to invest your initial capital in this asset. Consider two alternatives:

1.  (Constant Capital Allocation) Every day you allocate the same amount of capital to the risky asset. This approach is consistent with solving an MVO problem in each period. The problem faced in every period is
    `(13.1) max_w ( (1-p) log(1-wr) + p log(1+wr) ) ⇒ w* = (2p-1)/r = μ/r²`
    where `w` is the net amount allocated to the risky asset and is independent of the period.

2.  (Static Allocation) On day 0, you allocate a fraction `x` of your capital to the risky asset, and then you let it run. This is consistent with solving an MVO problem once, and letting it run.

3.  (Dynamic Allocation) Every day you allocate a fraction `x` of your capital on that day to the risky asset. We have no motivation for this (yet). The intuition, however, is that it seems reasonable to have a volatility proportional to the available capital in each period. The ratio of the strategy’s return to the available capital is `x r_t`. This is consistent with solving an MVO problem in each period, but with constant `x`.

Figure 13.1 shows the cumulative returns under the three approaches. The constant capital allocation shows low growth. Independently, `x = 1/2`, the static allocation has poor terminal wealth even though the risky asset has positive expected return. Lastly, `x = 1/2`, the dynamic allocation exhibits a variety of behaviors. The slope of the curve (not seen) is not monotonically increasing with `x`. A risk-adjusted performance is good for low values of `x`, but the average returns are low. The most profitable strategy corresponds to `x = 1/2`. Higher values detract from performance.

(Right Sidebar Text from Page 551)
**Figure 13.1:** Cumulative returns under the dynamic and static policies. All the curves are based on the same realization of returns of the risky asset. The returns are plotted on a logarithmic scale.

What is remarkable is that, for each strategy, and for each period, the Sharpe Ratio is identical and equal to `1/2`, because in each period the portfolio, being the combination of a risk-free asset and a risky asset, is mean-variance efficient. This simple example reveals well the subtleties of the Sharpe Ratio as a performance measure. It is an instantaneous measure of performance, and thus abstracting away any complication due to performance measurement, and we obtained the same value for all the strategies in our example. Yet, the behavior of the cumulative returns differs wildly among the strategies. We can interpret this finding in two ways. First, the commonly defined ex-post Sharpe Ratio (realized return divided by realized standard deviation in a single period), is a measure of single period, risk-adjusted performance. Averaging the Sharpe Ratio over the life of a strategy can give us a better estimate of `μ`, but it is not telling us much about the risk-adjusted performance of the strategy over its lifetime. If we instead use the cumulative returns over the strategy lifetime as a measure of performance, and ignore risk considerations—then the dynamic strategy with `x = 1/2` would be the clear favorite.

A second observation is that skill alone, defined as the ability to select a high-Sharpe portfolio in any given period, is necessary but not sufficient to be a successful investor. The size of the overall portfolio over time plays a major role in the long term. Yet, this topic does not receive much attention among academics or practitioners.

To understand where the value `x* = 1/2` comes from, let `r_t(x)` be the random return of the dynamic allocation strategy in period `t`. It is
`(13.2) 1 + r_t(x) = { 1+x, p=1/2; 1-x, p=1/2 }`
The total return of the strategy is `Π_{t=1}^T (1 + r_t(x))`. The average growth rate of the strategy `g_T(x)` is such that
`(13.3) Π_{t=1}^T (1 + r_t(x)) = exp(T g_T(x))`
where `g_T(x) := (1/T) Σ_{t=1}^T log(1 + r_t(x))`
For a fixed number of periods `T`, if we wanted to maximize the expected growth rate of the strategy, we would solve the problem `max_x g_T(x)`. As `T → ∞`, because returns are iid `r_t`, by the Law of Large Numbers,
`g_T(x) → E[log(1 + r_t(x))]`
a.s. The solution to the problem
`(13.4) max_x E[log(1 + r_t(x))]`
is asymptotically equivalent to maximizing the expectation (13.3). The objective function is maximized when the investment fraction `x` is equal to `1/2` (see Figure 13.3 and (13.8)).

**Figure 13.3:** Expected value of the log of the single-period growth, which is maximized at `x* = 1/2`.

(Right Sidebar Text from Page 555)
**Figure 13.2:** (a) Time series of cumulative returns for different fractions of the capital invested in the U.S. market benchmark (cap-weighted average of NYSE, AMEX, NASDAQ) from CRSP database. Monthly excess returns of the benchmark for the period February 1926–March 2018 are from Ken French’s Data Library site. (b) Cumulative returns as a function of the fraction invested in the U.S. market benchmark. The optimal Kelly fraction under the two approximations (Equations (13.9) and (13.10)) is `x* = 1.66` and `x* = 2.2`.

From Figure 13.2, it appears that this strategy performs decidedly better than other strategies with lower investment fractions. By simulation (or from Figure 13.1) one can see that any strategy in which `x > 1` performs worse: this corresponds to borrowing money to invest in the risky asset. Summing up, it appears that the Kelly criterion has maximized the long-term compounded growth rate of wealth by maximizing the expected growth rate, which is mathematically equivalent to maximizing an expected utility with a logarithmic utility function.

Let’s work out in detail an important example.

**Example 13.1 (The Kelly Allocation to a Single Security)**
We have only two assets: a risk-free asset and a risky asset. Let the excess return of the risky asset be `r` with mean `μ` and variance `σ²`. One way to interpret this asset in real-world application is as a portfolio manager to which we want to allocate capital. If we want to maximize the expected growth of the portfolio, then we would solve the problem
`(13.4) max_x g(x) := max_x E[log(1 + rx)]`
In addition to an exact numerical solution, we also produce an approximate solution based on the quadratic approximation of the logarithm:
`(13.5) log(1+x) ≈ x - x²/2 + O(x³)`
`(13.6) max_x E[log(1+rx)] ≈ max_x (μx - (1/2)(σ² + μ²)x²)`
and another approximation, in which we assume `μ² ≪ σ²`:
`(13.7) max_x E[log(1+rx)] ≈ max_x μx - (1/2)σ²x²`
from which
`(13.8) (exact result): x* = argmax E[log(1+rx)]`
`g(x*) = E[log(1+rx*)]`
`(13.9) (quadratic approximation): x* ≈ μ / (σ² + μ²)`
`g(x*) ≈ (1/2) μ² / (σ² + μ²) + 1`
`(assuming that) μ² ≪ σ²: x* ≈ μ/σ²`
`g(x*) ≈ (1/2) SR² + 1`
This approximate result is reliable when the typical fluctuations of `xr` are smaller than 1. A heuristic is to require that the volatility of `x*r` be smaller than 1: `|x*σ| < 1`, or `|SR| < 1`.

Let us consider in some more detail the accuracy of the approximations `x*₁, x*₂`. We consider daily Gaussian returns. The approximation `x*₁` is accurate relative error
`|x*₁ - x*|/x* ≤ 2%`
for daily Sharpe Ratios up to `0.1`, i.e., annualized Sharpe Ratio of `sqrt(252) * 0.1 ≈ 1.6`. The approximation `x*₂` is accurate (relative error
`|x*₂ - x*|/x* ≤ 2%`)
for daily Sharpe Ratios up to `0.02`, i.e., annualized Sharpe Ratio up to `0.3`. The crucial assumption in these calculations is the rebalancing interval. If we rebalance at shorter horizons, the volatility of returns is then smaller, and the quadratic approximation is more accurate. If we rebalanced at shorter horizons, then daily, the approximation would hold for higher values of the annualized Sharpe Ratio.

(Right Sidebar Text from Page 558)
Another way to read this result is that the optimal ratio between dollar volatility and capital is equal to the Sharpe Ratio. Hence the (dollar volatility)/(capital) ratio is
`(13.10) (dollar volatility) / (capital) = (W₀x*)σ / W₀ = SR`
Note that the formulas above hold for volatilities and Sharpe Ratios measured at the rebalancing horizon. For example, if we deploy `x*` of capital to a strategy with annualized Sharpe Ratio of `s`, then we must deploy approximately `x*s` of dollar volatility to run a Kelly-optimal strategy.

Another observation: according to Equation (13.9), the Kelly-optimal expected return is
`(13.11) (expected strategy return) = x*μ = SR²`
`(13.12) (dollar volatility) / (capital) = SR²`

**Example 13.2 (The Kelly Allocation to the U.S. Market)**
We can specialize the analysis above to the important case in which the risky asset is the U.S. market benchmark. This asset is available to retail investors in the form of low-management fees mutual funds (ETFs), both of which track the U.S. market accurately. For the U.S. markets are also available to institutional investors. The S&P500 index is a reasonable proxy for the S&P500 returns. The S&P500 index is a function of the underlying S&P500 returns. It has a Sharpe Ratio of `0.45` and an annualized volatility of `15%`. Based on the observed realization of the historical daily returns, the optimal return is `x*₁ = 1.66` (Equation (13.9)) and `x*₂ = 2.2` (Equation (13.10)). A fraction of `x* = 2.2` means that we borrow money to invest in the market. If we assume that we can borrow at the risk-free rate, then it would be optimal to leverage our capital. In practice, there are borrowing constraints and the fee behavior of a Kelly strategy also drawbacks: the historical PnL from a large `x` shows large drawdowns, which makes the overall fraction `x` have a very dramatic impact on capital appreciation.

The S&P500 example shows the attractive features, but also the drawbacks of Kelly strategies.

**Example 13.3 (Sizing a Bet)**
Consider a bet with a binary outcome: if we invest `$x`, we receive a payout equal to `r_w > 0` with probability `p` and `-r_l < 0` with probability `q = 1-p`. The optimization problem is
`max_x p log(1+xr_w) + q log(1-xr_l)`
`p r_w / (1+x*r_w) - q r_l / (1-x*r_l) = 0` (first-order condition)
`⇒ x* = p/r_l - q/r_w`
Introduce the win-loss ratio `ρ = q/p`, and the winning skew `s_w = r_w/r_l`:
`x* = (1/r_l) (1 - ρ/s_w)` (win-loss ratio) (winning skew)
The higher the win-loss ratio and the winning skew, the higher the size of the bet. If both win-loss ratio and winning skew are smaller than one, the optimal size can be negative.
We close this section with a more advanced example.

**Example 13.4 (Optimal Strategy for a Geometric Brownian Motion)**
Consider a simple risk-adjusted excess return `r` in a single period and rebalance the same period, which is then driven by `dW_t = x(t)μ dt + x(t)σ dW_t`. The strategy has initial value `W_0` and reliable equality `x = E[r] / (σ² + E[r]²)`. We start with capital `W_0`. Over the interval `[0,T]`, `x(t) = x_0`, the continuous-time process governing the capital accumulation at time `t` is given by
`dW_t = x(t)μ W_t dt + x(t)σ W_t dB_t`
Here, `B_t` is standard Brownian process. It can be shown that the distribution at time `t` is given by
`W_T = W_0 exp [ ∫₀^T (x(t)μ - (1/2)x(t)²σ²) dt + ∫₀^T x(t)σ dB_t ]`
Hence
`g_T(x) = (1/T) ∫₀^T (x(t)μ - (1/2)x(t)²σ²) dt + (1/T) ∫₀^T x(t)σ dB_t`
For `T → ∞`, the second integral converges to zero a.s. (for intuition, `B_t` scales like `√T`). The first integral is maximized when the integrand is maximized for all `t`, which occurs when
`x(t) = μ/σ² = SR / σ`

(Right Sidebar Text from Page 561)
We have recovered Equation (13.13).
In our presentation, we have ignored two important features:
*   The Sharpe Ratio is a decreasing function of the capital `x` allocated to the active strategy. Modeling this dependency explicitly is challenging, and to my knowledge there is no analysis where `μ` is a function of `x`.
*   The formula `x* = SR/σ` (or does not hold any longer. At the very least, we should acknowledge this dependency in the ideal formula, and allow for the allocation to depend on the available capital `W_t`).
*   There are transaction costs even in the case of a static `x*`. Whenever the active strategy has a positive PnL, in order to maintain a fraction `x*` in the active strategy, we need to partially trade out of the policy and allocate to the risk-free asset. The dollar amount we need to trade in order to rebalance the fraction `x*` is
    `W_T*(1+r_1) - x* = W_T r_1`
    `W_T*(1-r_2) + x* = W_T r_2`
    Trading costs are super-linear in `Δx`, and therefore super-linear in `W_T`, and become dominant as wealth grows. HIC SVNT LEONES.

The next section is devoted to describing the attractive mathematical properties of Kelly strategies.

---

**13.2 Mathematical Properties**

We now generalize the case in which we can choose to invest across a portfolio of `n` strategies `dp_i`, and the associated returns `r_i(θ)` are independent of `x_i` for all `i = 1, ..., d`. These results were proved first by Breiman (1961) and Dubins and Savage (1965) for iid returns, and the results that follow apply to this case. Some of them have also been established for dependent random variables; see Algoet and Cover (1988).

1.  Let `X_t` be the cumulative returns of the Kelly strategy, and `Y_t` of an alternative strategy with lower expected growth rate.
    The first property is that the Kelly strategy grows faster than any other strategy. Let `X_t, Y_t` be the cumulative returns of the Kelly strategy and alternative strategy respectively. Then, with probability 1,
    `(13.13) Y_t/X_t → 0` as `t → ∞`
2.  The second property concerns the long-term growth of a strategy based on expected value of its log returns. Let
    `g = E[log(1+r_t)]`
    and `X_t` the associated cumulative return process. Then, with probability 1,
    `g > 0 ⇒ X_t → ∞`
    `g < 0 ⇒ X_t → 0`
    `g = 0 ⇒ lim sup_t X_t = ∞, lim inf_t X_t = -∞`
3.  The expected time to reach capital level `C*` is equal to `log C* / g` in the limit `C* → ∞`, and the drawdown time is Kelly strategy.

**Insight 13.1: The Intuition behind Kelly Strategies**
The Kelly criterion has several intuitive and attractive features:
*   Goal: The allocation strategy achieves the highest long-term capital growth.
*   Strategy: The optimal strategy is simple, since it allocates a constant fraction of total capital to the risky strategy.
*   Lower and Upper Bounds on Risky Allocation: The fraction of capital allocated to a risky strategy increases with high `μ` and `σ` (up to `g > 0`) and is around `μ/σ²`.
*   Sharpe Proportional: To a first approximation, the optimal fraction of invested capital and the volatility/capital ratio are proportional to the Sharpe Ratio of the strategy.

(Right Sidebar Text from Page 564)
What these results say is that a Kelly strategy has many very desirable features. In the long run, it beats almost surely any other strategy that has a different expected growth rate. It reaches a certain capital level faster than any other strategy, and the approximate time needed to reach this return can be expressed as a function of `g`. Finally, a positive expected growth rate is necessary and sufficient condition for any strategy to have a growing cumulative return over time.

What the results don’t say is that a Kelly strategy is maximizing the Sharpe Ratio, even if we were able to compute it exactly from knowledge of the true expected returns and volatility of the strategy. There is no guarantee that any Kelly strategy will outperform a drawdown, which can be severe, as seen in the simulations above. In Example 13.2, as the fraction `x` invested increases from 0 to the growth-maximizing level `g**`, both the growth rate and the size of the drawdowns increase with `x`. The scale of the y-axis is logarithmic. The existence of this trade-off is therefore properly captured by the simulations. As we move from the optimal level, the average returns (as expected) and the drawdowns increase further. For fractions of the invested wealth lower than `g**`, there is a trade-off between expected log returns and volatility of the log returns: we get lower returns, in exchange for lower risk. We explore this trade-off next.

---

**13.3 The Fractional Kelly Strategy**

The fractional Kelly strategy consists of investing `x*` in a strategy with iid return `P_t`, a fraction `x_{frac}` of the available capital (smaller than `x*`) such that it will satisfy that
`E[log(1+r_t x_{frac})] > 1`
It can be interpreted in several ways:
*   Combination of risk-free asset and full Kelly. Fractional Kelly is a combination of two investments: a risk-free asset and the full Kelly strategy.
*   The most common interpretation is a utility-based one. In a series of analyses pursued by MacLean et al. (1992) they show (MacLean et al., 2010, 2012) that the fractional Kelly strategy does indeed trade off growth for security. Assume, for example, that in each period we are considering a percentage return `r` greater than the risk-free rate `r_f`. We would be willing to accept a maximum drawdown per period that we can accept. From Equation (13.10), `x* = SR/σ`. We choose the minimum of the
    `x* = min(p/σ, SR)`

**Example 13.5:**
We deploy $/X of capital, and have a Sharpe Ratio of `s`. The strategy has an average annual return of `5%`. We can lose at most `p%` of capital in a year. We find that
`3 × (weekly dollar volatility) = 0.01 × (capital)`
i.e.,
`(weekly dollar volatility) / (capital) = 0.01/3`
`p = (weekly dollar volatility) / (capital) = 0.01/3`
So that `p/σ = 0.11`. This is smaller than the weekly Sharpe Ratio `2/√52 ≈ 0.27`, which corresponds to the optimal Kelly position.
Higher risk aversion. We start from Equation (13.4), which approximates the log-objective function with a near-quadratic one. The quadratic penalty:
`max_x μx - λ/2 (σ² + μ²)x², θ > 1`
The optimization point is
`x_{frac}^* = μ / (λ(σ² + μ²))`
Fractional Kelly is thus a modified Kelly strategy for investors who are more risk-averse than logarithmic utility would suggest.

Parameter uncertainty. Thorp (2006) makes the case that uncertainty about the properties of returns should result in fractional Kelly. Indeed, being wrong can have terrible consequences. Imagine, for example, that we have a strategy with a volatility of `15%`, and an estimate of Sharpe Ratio equal to `0.45`. The Kelly fraction is `x* = 2`. We over-leverage the portfolio by a factor of 2. We can perform a Monte Carlo simulation based on the returns of the market.
We have introduced parameter uncertainty already in Section 10.3.2, in the context of MVO. Here we model parameter uncertainty as follows. Percentage returns are `r_t = r(ω_t, θ_t)`, where `P_t` is a function of a random variable `ω_t` (the sample path of returns) and of a random variable `θ_t` (the quality of the estimate). We have a sample of `N` pairs of `(ω_t, θ_t)` drawn from a probability measure `P_N`. And `θ_t` is `θ` in random parameter setting values in a set `Θ` with probability measure `P_Θ`. The interpretation is that in every period we have a noisy estimate of the true parameter `θ* = E_Θ[θ_t]`. We make the crucial assumption that `ω_t` and `θ_t` are independent. Also, `ω_t` and `θ_t` are independent. Also `ω_t` and `θ_t` are independent. Also `ω_t` and `θ_t` are independent.
The expectation is taken with respect to the random variables `ω ~ P_ω` and `θ ~ P_Θ`. We want to maximize `g(x)`. The first-order condition is `g'(x) = 0`.
`g(x) = (1/T) Σ_{t=1}^T log(1+xr(ω_t, θ_t))`
`g(x) = lim_{T→∞} g_T(x)`
`= E_{ω,θ} [log(1+xr(ω,θ))]` a.s.
`g'(x) = E_{ω,θ} [ r(ω,θ) / (1+xr(ω,θ)) ]`
`g'(x_{concert}^*) = 0`
As a function of `x`, the function `h(τ(θ)) = E_ω[log(1+xτ(ω,θ))]` is increasing and strictly concave. Then it follows that, as a function of `θ`, `h(τ(θ))` is concave, because
`∂²h(τ(θ))/∂θ² = h''(τ(θ))τ'(θ)² + h'(τ(θ))τ''(θ) ≤ 0`
By Jensen’s inequality,
`E_θ [ r(ω,θ) / (1+xτ(ω,θ)) ] ≤ E_θ[r(ω,θ)] / (1+xE_θ[τ(ω,θ)])`
And therefore, taking expectations over `ω`,
`g'(x) = E_ω E_θ [ r(ω,θ) / (1+xτ(ω,θ)) ] ≤ E_ω [ E_θ[r(ω,θ)] / (1+xE_θ[τ(ω,θ)]) ] = g_0(x)`
The function `g_0(x)` on the left-hand side is the derivative of the expected log return in the absence of parameter uncertainty. The function `g(x)` on the right-hand side is the derivative of the expected log return when the parameter is known. It follows that `x_{concert}^* ≤ x_{known}^*`. Figure 13.4 visually illustrates the location of the two solutions.

**Figure 13.4:** The optimal Kelly size in the presence of parameter uncertainty is always smaller than the optimal size when parameters are known.

Let us consider two examples that have some general application: uncertainty about a strategy’s expected return and about its variance.

**Example 13.6 (Strategy with Uncertain Expected Return)**
Let `r = θ + ε`, where `E_ω[ε] = 0`, `var_ω[ε] = σ²`, `E_θ[θ] = μ_0` and `var_θ[θ] = σ_θ²`.
`E[log(1+rx)] ≈ μ_0 x - (1/2) (σ² + μ_0² + σ_θ²) x²`
so that the Kelly fraction is
`x_{frac}^* ≈ μ_0 / (σ² + μ_0² + σ_θ²)`
Let us use the S&P500 market estimates from previous examples: `σ = 0.15`, `μ_0 = 0.08`, and `σ_θ = 0.04`. We get `x*_{frac} = 1.81`, compared to an estimate of `1.66` in absence of estimation error.

**Example 13.7 (Strategy with Uncertain Volatility)**
Let `r = μ + ε`, where `E_ω[ε] = 0`, `var_ω[ε] = θ` and `E_θ[θ] = σ_0²` and `var_θ[θ] = σ_θ⁴`. We assume that `E_θ[√θ]` and `θ` are independent. Also in this case `E_θ[√θ]² / σ_0² = 0`.
`E[log(1+rx)] ≈ μx - (1/2) (E_θ[θ] + μ²) x²`
`= μx - (1/2) (σ_0² + σ_θ² + μ²) x²`
so that the Kelly fraction is again
`x_{frac}^* ≈ μ / (σ_0² + σ_θ² + μ²)`
Let us use the S&P500 market estimates from previous examples: `σ_0 = 0.15`, `μ = 0.08`, and `σ_θ = 0.1` for the market return. We get `x*_{frac} = 1.04`, compared to an estimate of `1.66` in the absence of estimation error.

**Insight 13.2: All reasonable investors use fractional Kelly without knowing**
All reasonable investors allocate capital to a risky strategy so that the volatility/capital ratio is constant, or slowly varying. This is because the Kelly criterion is the only one that, in the long run, allows them to allocate as much capital as possible, compatibly with the drawdowns that their investors can bear. In a series of papers (Part VI of MacLean et al. (2010)) Ziemba (jointly collects the contributions on this subject).
Ziemba and coauthors provide anecdotal evidence that successful investors follow Kelly allocations. The likely reason is not that they are aware of the Kelly criterion, but rather that they use the simple common-sense (inspired) heuristic, which turns out to be equal to fractional Kelly.

---

**13.4 Fractional Kelly and Drawdown Control**

In an influential paper, Grossman and Zhou (1993) address a question related to the issue of finding an optimal trading strategy and of controlled growth/optimal consumption. In the Grossman–Zhou framework (and henceforth), the investor wants to maximize the long-term growth and with probability one avoids reaching a drawdown threshold. As formulated in their original paper, the model only considers a risk-free asset and a risky one. However, it generates different results than pure Kelly. We define the high watermark of the wealth process `W_t` in order to formulate the policy, we define the high watermark of the wealth process `W_t`:
`M_k = max_{0≤s≤k} W_s`
Let `d_k` be the current drawdown percentage from the high watermark: `d_k = 1 - W_k/M_k`. Let the maximum allowed percentage drawdown be `D`. The optimal policy gives the optimal fraction invested in the risky asset and is given by
`(13.15) f_k = (μ/σ²) ( (1-D) / (1-d_k) )`
This policy is elegant and intuitive. For some intuition, fix first `D = 1`, i.e., we can tolerate any drawdown. Then the strategy is to use the unconstrained Kelly allocation `f_k = μ/σ²`. If `d_k = 0`, i.e., then the optimal policy is to invest a fraction `f* (1-D)` where `f*` is the high watermark `M_k = W_k`. This means that we are more prudent than in the simple Kelly scenario, and we are more prudent if our threshold `D` is conservative, a “dynamic fractional Kelly” with maximum drawdown. Moreover, we decrease the investment fraction as we approach the drawdown threshold, and we liquidate the risky asset. Figure 13.5 shows the optimal fraction as a function of the threshold. The reduction rate is nearly constant over the range of allowed drawdowns.

(Right Sidebar Text from Page 576)
**Figure 13.5:** Percentage reduction factor `(1-D)/(1-d_k)`.

**Insight 13.3: Modulating volatility reduces the Sharpe Ratio**
The GZ criterion changes expected returns, and hence reduces the Sharpe Ratio. Over time, changing the volatility of a strategy when the volatility is independent of the expected return of the strategy is Sharpe reducing. To gain intuition about this fact, consider the simple example of a strategy with Sharpe Ratio equal to `s`. Half of the year we deploy it at volatility `σ₁/2`, and half of the year at volatility `(3/2)σ₁`. The expected PnL for the entire year is
`((1/2)sσ₁ + (3/2)sσ₁) / 2 = sσ₁`
The annualized volatility is
`√((σ₁/2)² + (3/2)²σ₁²) / 2 = √(11/8)σ₁`
The Sharpe Ratio is `√(8/11)s`. Compare to the case in which we had kept the volatility constant at the value `√(11/8)s`. The Sharpe Ratio would have been `s`. This example shows that drawdown control doesn’t come for free.

The strategy is a continuous version of the stop-loss policies employed by many hedge funds and successful investors. In the presence of a large drawdown, a portfolio manager operating autonomously within the fund is required to partially or completely liquidate her portfolio. The strategy has many interpretations. One interpretation of a stop-loss policy is a real insurance policy on the strategy itself. View the policy as a synthetic put option, whose price is `W_0`. Imagine that we hold an out-of-the-money put.

To understand the trade-offs between optimizing for variance control and optimizing for drawdown control, it is useful to compare the GZ and fractional Kelly strategies in a numerical example. Specifically, we consider the case of a risky asset with independent, identically distributed returns. Its expected daily return is `0.08%` and its daily volatility is `1%`, corresponding to a Sharpe Ratio of `1.23`. The two strategies are parameterized by the Kelly fraction and the drawdown threshold, respectively `f_k, D`.
`(13.16) f_k(D) = (μ/σ²) P` (fractional Kelly)
`(13.17) f_k(D) = (μ/σ²) ( (1-D) / (1-d_k) )` (Grossman-Zhou)
with `p ∈ (0,1), d_k ∈ (0,1)`
I then simulate the performance of the two strategies over a one-year period (i.e., 25,200 days) and compare the realized volatility and the maximum drawdown for strategies having the same expected log-return. Figure 13.6 shows the results. As expected, the fractional Kelly strategy has a better profile than GZ in the mean-variance sense, and is superior up to the maximum allowed drawdown (10%). A numerical optimization for reduction of GZ seems more desirable than the associated increase in volatility.

(Right Sidebar Text from Page 580)
**Figure 13.6:** Comparison of fractional Kelly and Grossman–Zhou strategies. Both strategies’ performance measures are estimated over the same sequence of 25,200 returns, but with different parameters `P_k`, `D_k`. (a) Standard deviation of daily log-returns versus mean log-return. (b) Maximum drawdown.

For example, consider a max tolerated drawdown of `10%`. GZ achieves an average daily return of approximately `0.025%`, while fractional Kelly achieves an average daily return of `0.035%`, a `20%` increase. More importantly, GZ controls the maximum drawdown ex-ante, with probability one and independently of misspecification of the problem. In the fractional Kelly approach, even if we provide a maximum drawdown threshold, this threshold is not guaranteed to be met. The optimization problem assumes this bound will be respected as well. These considerations suggest that the GZ strategy may be preferable. There is an important qualification to this statement. Throughout this chapter, we have ignored the role played by transaction costs. As mentioned above, a pure fractional Kelly policy implies the continuous rebalancing of capital from a risky asset to a risk-free one. GZ, in the event of a drawdown sufficient to trigger itself in wiping over time, sometimes very rapidly in the fraction allocated to the strategy, then we may force a complete liquidation of the risky asset when we reach the threshold. This in turn may allow for a large reduction in transaction costs. Beyond the scope of this book, we would then need to extend to the case of transaction costs which, in the absence of analytical results, may only be tractable with numerical experiments. These objections notwithstanding, GZ is a useful heuristic that can be used as an overlay to a Kelly-like strategy.

---

**The Takeaways**

1.  The policy that generates the highest capital growth in the long run is the Kelly criterion.
2.  The Kelly criterion prescribes that we allocate a constant fraction of our capital over time to our active strategy that is equal to the Sharpe Ratio of our strategy divided by its annualized volatility.
3.  The Kelly criterion has the undesirable property of incurring large drawdowns over time.
4.  To alleviate this problem, we can adopt the fractional Kelly criterion, which allocates a constant, smaller fraction of capital to the active strategy. It trades off growth for higher security.
5.  Parameter uncertainty is another way to justify the fractional Kelly criterion.
6.  It is further possible to modify the Kelly criterion so that the fraction of capital is a function of the maximum tolerable drawdown, and this fraction is linearly decreasing as a function of the drawdown size. Once the strategy hits its loss limit, it is liquidated. A deterministic guarantee on the experienced drawdown.
7.  Many successful investors naturally follow a fractional Kelly with drawdown control.
8.  All of these simple strategies are valid in the absence of transaction costs. They need to be simulated and calibrated in real-world applications, in order to account for such costs.

---

**Notes**

1.  For an intuitive treatment of the Kelly criterion, with plenty of examples and applications, see Haghani and White (2022). Marku’s Kurtis’ blog “Outcast Beta” is also bridging the world of application and the mostly academic literature.
2.  A. Damodaran maintains a page (home.stern.nyu.edu/~adamodar/) with S&P500 and Treasury returns.
3.  Before January 1931, the S&P500 had 90 components. The returns for 1926–1930 are from S&P90 index.
4.  For the other properties of the Kelly strategy, See MacLean et al. (2010a) for a review (or Part IV of the book MacLean et al. (2010b) for a diversity of views.
5.  The relationship between this mean-variance approximation and power utility function `u(x) = x^(1-γ)/(1-γ)`, for `γ > 0`, is explored by Pulley (1981).
   

Okay, here is the Markdown compilation for Chapter 14.
     Everand

