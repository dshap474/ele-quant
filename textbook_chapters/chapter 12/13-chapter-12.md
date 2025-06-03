**Chapter 12**
**Hedging**

**The Questions**

1.  What is the purpose of hedging in a portfolio, and what are the different types of hedges available?
2.  How can a simple hedging model with no transaction costs or parameter uncertainty apply to real-world portfolio management?
3.  How does the inclusion of parameter uncertainty affect optimal hedging strategies?
4.  What are the implications of execution costs in multi-period hedging?
5.  How can factor models be used in constructing hedging strategies, especially when managing factor exposures?
6.  What scenarios is it beneficial to hedge with FMPs, liquid assets, or futures, and how do these impact portfolio risk and return?

Hedging is the process of reducing the risk of a pre-existing portfolio by means of augmenting the portfolio with additional investments, whose returns are negatively correlated to the existing portfolio. The most common forms of hedging are market hedging and exposure hedging. In these cases, we use an instrument or a combination of instruments to partially manage the risk. The first is hedging by means of Factor-Mimicking Portfolios (FMPs) obtained from a fundamental factor model. The second one is hedging by means of a future or liquid asset capturing equity and non-equity risk. This includes energy and interest rate futures, bond futures, ETFs and ETNs (exchange-traded (ET) notes, i.e., debt). The first application of interest is, e.g., XLE, an ETF tracking the stocks in the technology sector), or style (e.g., MTUM, an ETF reproducing the behavior of the momentum factor) risk. The last application of interest is the creation of thematic tradeable baskets by banks. One can buy these baskets to hedge or speculate on political risk (e.g., elections) or economic risk (e.g., G7 vs. China) thematic exposure using investible baskets.

This chapter is broadly organized into three sections. The first part covers naive hedging. There are no effects in this section: no transaction costs, no parameter uncertainty, and a single period. Yet, such a simple model is still widely used in many applications. In the second part we explore the impact of parameter error and how it affects optimal hedging. Lastly, we look at multi-period hedging in the presence of execution costs.

---

**12.1 Toy Story**

In its simplest form, we have the following ingredients:
*   We have two decision dates `t₀, t₁`, and one realized return between them. We make investment decisions at `t₀` and observe realized returns at `t₁`.
*   We have two assets, which we denote core and hedge, with returns `r_c, r_h`, expected returns `μ_c, μ_h = 0`, volatilities `σ_c, σ_h > 0` and return correlation between the two equal to `ρ_ch`.

We decide the size of the hedging instrument in order to maximize the Sharpe Ratio of the combined portfolio. We already know how to solve this problem: it is the two-asset Mean-Variance Optimization (MVO) instance we saw in Section 10.1. In that problem, we decided the optimal positions of both assets, not a major difference. The MVO optimization problem
`max_{x_c, x_h} μ_c x_c / √(x_c²σ_c² + x_h²σ_h² + 2ρ_ch σ_c σ_h x_c x_h)`
has solution
`(12.1) x_c* = ρ_ch σ_h / σ_c x_h*`
`(12.2) x_h* = - (ρ_ch σ_c / σ_h) x_c* = -β(r_c, r_h) x_c*`
The ratio `|x_h* / x_c*|` is the optimal hedge ratio and is equal to the beta of the core portfolio’s return to the hedging portfolio’s return.

The unhedged variance is `x_c*²σ_c²`; after hedging it is `(1 - ρ_ch²)x_c*²σ_c²`. The improvement in Sharpe Ratio is equal to the improvement in volatility:
`(12.3) SR(hedged) / SR(native) = 1 / √(1 - ρ_ch²)`

The parameter `β` is estimated either via time-series regression or by using a return covariance matrix, such as one supplied by a factor model. Define `w_c` the core and hedge portfolios, the model beta is
`(12.4) β(r_c, r_h) = w_cᵀΩ_f w_h / (w_hᵀΩ_f w_h)`
From that Equation (12.2) gives the relative size of the hedge, and Equation (12.3) the improvement in Sharpe Ratio from hedging.

In their simplicity, Equations (12.1)-(12.3) are applied widely. A typical application involves the use of a single hedging instrument that is very liquid and inexpensive to trade, and whose expected return is negligible compared to that of the core portfolio. Examples are E-mini S&P500 futures and the SPY, VIX and VOO ETFs, which also track the S&P500 index. We perform intraday or end-of-day hedging in order to remove the associated risk.

**Exercise 12.1 (Comparing a Market FMP and Benchmark as Hedging Instruments).**
1.  For this exercise, you will need two portfolios (maybe for a live strategy), an equity factor model, and the weights of several benchmarks.
    Compare the risk decompositions of the benchmark and of the market FMP. What is their idiosyncratic variance as a percentage of total variance? What are their exposures (this applies to benchmarks only)?
2.  Now, hedge the test portfolios using the FMP and the benchmark. What is the reduction in factor variance in the two cases? What is the increase in non-market, i.e., idiosyncratic, risk? Are these partially hedged out? Are they material?
Perform the analysis over a number of years to verify whether the findings are stable.

**Procedure 12.1: Simple single-asset hedging**
**Inputs:** Core portfolio NAV `x_c`, with returns `r_c`.
Hedge asset with returns `r_h`. Parameter `ρ_ch = ρ(r_c, r_h)` estimated by means of time-series regression, or of an asset covariance matrix and Equation (12.4).
**Output:** Hedge NAV `x_h* = -ρ_ch (σ_c/σ_h) x_c`.

Hedging in this specific instance rests on several implicit and explicit assumptions:
*   We assume that the beta of the core portfolio to the hedging instrument can be estimated accurately.
*   We assume that there is a single hedging instrument.
*   We assume that trading costs are negligible.
*   We assume that the hedging instrument has negligible expected return.
In the remainder of this chapter we re-examine these hypotheses and relax them.

---

**12.2 Factor Hedging**

**12.2.1 The General Case**

We have recurrent use of factor models in this book, and unsurprisingly they make their hedging debut as portfolio performance enhancement tools that take into account the portfolio risk arising from factor exposures and idiosyncratic risk, and generate a portfolio that meets our investment goal. In practice, there are situations in which this is not possible. An important instance is when the core portfolio is the outcome of a portfolio construction process outside of our control. For example, an investment group may have independently developed a portfolio of long-term trading stocks based on their fundamental outlook. The last of these individual portfolios constitutes a core portfolio that is not optimized, and that exhibits undesired systematic risk. In this case, the hedging process takes `w_c` as an input, and seeks to reduce the unwanted risk from factor exposures. We use a factor model with `K` fundamental factors, as in Chapter 7. They are the columns of matrix `F`, and have unit exposure to factor `j`. One way to hedge factor risk would be Procedure 12.2.

**Procedure 12.2: A simple factor hedging procedure**
1.  Compute the core portfolio factor exposure `b_c = Bᵀw_c`.
2.  “Hedge out” the core exposure by finding an amount of factor exposure `-b_c`. We do this by buying a hedge portfolio `-B⁻¹b_c`.

We have achieved zero factor exposure. The solution is simple, elegant, and unfortunately unrealistic. We have ignored two essential aspects of the hedging problem. First, factors have non-zero expected returns. Second, trading factors is expensive. However, we can change the formulation to include these modeling concerns. Let us begin with assuming that the factor-mimicking portfolio of the hedging strategy has zero alpha and zero trading costs. In Section 12.2.2, when the definition of alpha is orthogonal to alpha spanned. In formulas: the expected return of a portfolio is equal to `(α_cᵀ + μ_FᵀB)w`. Regarding the execution costs, we include them in the optimization formulation, using a mean-variance framework, as in Chapter 9. We maximize the expected return of the combined portfolio of a portfolio `w_c` from a starting portfolio `w_0` as `f(w_c)`. The trading cost of a portfolio is `p`. In a single-period setting, we then write the problem as
`(12.5) max_{w_h} α_cᵀ(w_c + w_h) + μ_FᵀBᵀ(w_c + w_h) - (1/2)(w_c + w_h)ᵀΩ(w_c + w_h) + f(w_c, w_h)`
`s.t. Bᵀ(w_c + w_h) = b*`
`α_cᵀ(w_c + w_h) = (expected returns)`
`w_c, w_h ∈ ℝ^n`

I leave it as an exercise to prove that if execution costs are zero, orthogonal and spanned alphas are zero, and factor portfolios have zero idiosyncratic variance, then of course we would hedge out exposure. Not a single one ofthese assumptions holds, and it is worth spending some time commenting on them.
*   Some of the factors do have zero expected returns, some don’t. Hedging them is in fact a trade-off: we accept losses because the gains in risk reduction are (hopefully) worth the losses.
*   The hedging portfolios may also have non-zero alpha orthogonal exposure. This must be taken into account, especially when alpha orthogonal is what determined what the profitability of the strategy depends on, more than our alpha spanned.
*   Even if we traded the pure FMPs of Procedure 7.2, we would add idiosyncratic risk to the core portfolio. This additional idiosyncratic risk reduces the benefits of factor risk reduction. The optimization formulation takes this into account. In fact, the following exercise asks you to derive the details and show that the optimal hedging is not equal to `-b_c`.

**Exercise 12.2:**
Assume that:
1.  factor portfolios have zero expected returns;
2.  we hedge using only factor portfolios;
3.  we have no transaction costs.
Prove that the optimal hedging policy is
`x* = -(BᵀΩ⁻¹B)⁻¹BᵀΩ⁻¹b_c - (BᵀΩ⁻¹B)⁻¹BᵀΩ⁻¹b_h`
Under what condition is the optimal hedging ratio smaller than the perfect factor neutralization of Procedure 12.2?
The solution is in the Appendix. Meanwhile, here is a much easier problem to get you started.

**Exercise 12.3:**
For simplicity, consider the case where asset returns are described by a one-factor model, and there is a hedging portfolio that has exposure to that factor. Starting with Equation (12.4), show that it is optimal not to hedge entirely the exposure of the core portfolio to that factor.

In the simplistic hedging procedures of the first part of this section, we did not have an investment objective, because volatility reduction was a pseudo-investment objective: no execution concerns, no expected factor returns, no idiosyncratic volatility increase. But reality is complicated. The parameter `ρ` quantifies our risk tolerance and determines where we want to be on the curve trading off volatility and expected return. This is a good thing. In practice, we should explore this trade-off and determine the optimal operating point.

In the special case of quadratic costs, Optimization Problem (12.5) can be rewritten as a multi-period optimization problem (see Chapter 11), using the formalism presented in Appendix A and specifically in Procedure 11.1.

**Exercise 12.4:**
Extend Optimization Problem (12.5) to the multiperiod setting. Discuss the implementation complexity and propose some simplifying assumptions.

---

**12.3 Hedging Tradeable Factors with Time Series**

A relatively common use case for hedging is the following. There are non-equity tradeable and liquid instruments that are associated with macroeconomic movements; for example, energy or metal commodity futures, or fixed-income futures. Because of their ability to capture broad macroeconomic themes and their liquidity, we would like to use these instruments for hedging. To fix our ideas further, consider the case of a portfolio of international stocks, and of gold. We decide to hedge the portfolio returns using West Texas Intermediate (WTI) crude oil futures. We assume that the energy portfolio is correlated to energy prices, and at the same time that the portfolio manager or the trading algorithm does not have a view on the future energy price movements. A possible approach is to estimate time-series betas `β_ch = ρ_ch σ_c / σ_h` and then hedge the core portfolio `w_c` using a hedge portfolio `w_h = -β_ch w_c`. This approach won’t work! Surprisingly, or more likely, in real-world instance, the realized risk of the hedged portfolio was worse than the realized risk of the core portfolio. This is somewhat counterintuitive. In this section, we aim to shed some light on hedging for this particular scenario.

As in the previous sections, we denote the return of the tradeable instrument `r_h` with variance `σ_h²`. We model the instrument returns `r_h = β_h r_m + η_h`, in other words, it has factor exposure to a given set of factors `r_m`, and idiosyncratic return `η_h`. We have a random estimation error with covariance matrix `Ω_h` and `β` is the vector of true betas. In order to see what could go wrong, let us hedge with the “optimal” hedge ratio:
`w_h* = - (β - η)ᵀw`
The covariance matrix, augmented with the hedging instrument, is
`Ω̃ = [[Ω_c, Ω_{ch}], [Ω_{hc}, Ω_h]]`
Let us compute the variance of the hedged portfolio:
`var(r_c + r_h*) = E_η[(w_c + w_h*)ᵀΩ̃(w_c + w_h*)]`
`= E_η[w_cᵀΩ_c w_c - 2w_cᵀΩ_{ch}(β-η)w_c + (β-η)ᵀw_cᵀΩ_h w_c(β-η)]`
The variance of the hedged portfolio exceeds the unhedged variance when
`w_cᵀΩ_h w_c > (βᵀw_c)²`
The left-hand side of the inequality is the squared estimation error of the portfolio beta. The right-hand side is the portfolio beta-related variance.

Between the non-hedged and the fully hedged portfolio, maybe there is a hedging level that improves on both. We consider the case where we apply a positive hedging shrinkage factor `γ*` to the optimal hedging `w_h* = -β_ch w_c`. The hedged portfolio is `w_c + γ*w_h* = (1 - γ*β_ch)w_c`, and hence we minimize the expected loss `E_L`. The formula for the variance is similar to the one we performed above:
`E[r_c + r_h*] = E_η[ (w_c + γ*w_h*)ᵀΩ̃(w_c + γ*w_h*) ]`
`= w_cᵀΩ_c w_c - 2γ*w_cᵀΩ_{ch}βw_c + γ*²βᵀw_cᵀΩ_h w_cβ`
which we minimize to find the optimal shrinkage factor and hedge ratio:
`(12.6) γ* = w_cᵀΩ_{ch}β / (βᵀΩ_h β)`
`(12.7) w_h* = -γ* βᵀw = - (w_cᵀΩ_{ch}β / (βᵀΩ_h β)) βᵀw`
Let’s sanity-check this formula.
*   The shrinkage factor `γ*` is independent of the units of the portfolio. If we measure the portfolio in cents or in dollars, we get the same value of `γ*`. Otherwise stated: if we hedge a portfolio 10 times the size of our portfolio, the hedge factor is unchanged, and the dollar hedge is 10 times the hedge of the original portfolio.
*   If there are no estimation errors in the betas, then `Ω_h = 0` and `γ* = 1`; we use the optimal hedge ratio.
*   The numerator is a weighted sum of the estimation errors. The larger the errors, the smaller the shrinkage factor.
*   The ratio `w_cᵀΩ_{ch}β / (βᵀΩ_h β)` can be loosely interpreted as the square of the aggregate noise-to-signal ratio of the betas. The higher the ratio, the smaller the scaling factor.
*   Consider the edge case where the true betas are all zero and errors are independent. Then `Ω_{ch} = 0` and the expected value of the denominator is
    `E[ (w_h*)ᵀΩ_h w_h* ] = w_h*ᵀΩ_h w_h*`
    In expectation, numerator and denominator are equal, and `γ* = 0`. On average, we do not hedge. This is the correct course of action.
In practice, we do the following steps:
1.  Estimate the time-series `β_ch` and its standard error `σ_β`. Define `Ω_h` as the diagonal matrix whose `j`-th term is `σ_β_j²`.
2.  Compute `γ*` using Equation (12.6).
3.  Buy `x_h* = -(γ* β_chᵀw_c)` of the hedge instrument. The lower bound on `γ*` is meant to avoid the situation where we hedge in the opposite direction.
4.  (optional) It is difficult to estimate the correlations between estimation errors, especially in periods of market stress. You can simulate their impact by assuming constant correlations between them and then defining
    `Ω_h = [[σ_β₁², ρσ_β₁σ_β₂, ..., ρσ_β₁σ_β_n], [ρσ_β₁σ_β₂, σ_β₂², ..., ρσ_β₂σ_β_n], ..., [ρσ_β₁σ_β_n, ..., σ_β_n²]]`
    and testing the sensitivity for different values of `ρ`. The hedging ratio decreases linearly as `ρ` increases.
5.  (simplifying Equations (12.6) and (12.7)) Assume that the terms `w_cᵀΩ_{ch}` are cross-sectionally uncorrelated with `w_cᵀβ`. Then,
    `w_cᵀΩ_{ch}w_c* = Σ_j w_{c,j}² σ_{β_j}²`
    `= E_η[ (w_cᵀβ)² ] + E_η[ (w_cᵀ(β̃-β))² ... ]` (This line seems to be a derivation step or an expansion)
    An analogous simplification occurs for the denominator. Then the formula for the optimal hedge ratio becomes
    `(12.8) γ* = E_η[ (w_cᵀβ)² ] / ( E_η[ (w_cᵀβ)² ] + E_η[ (w_cᵀ(β̃-β))² ] )`
    Higher standard errors `σ_β` imply greater shrinkage. Lower dollar exposure to the tradeable factor also means greater shrinkage. Finally, to simplify things dramatically, consider the case where all `β_j` are identical, and the portfolio is long only. The shrinkage factor simplifies to:
    `γ* = (β̃ E[r_c])² / ( (β̃ E[r_c])² + E[ (β̃-β)² E[r_c]² ] ) * H(w)`
    (squared noise-to-signal) (portfolio concentration)
    The ratio `H(w) = ||w||₁² / ||w||₂²`
    is a measure of portfolio concentration that has maximum diversification `H(w) = n` (n positions with identical NAV) and has `H(w) = 1/n`, while a maximally concentrated portfolio has all NAV concentrated in a single stock, so that `H(w) = 1`. The interpretation here is that the shrinkage factor is smaller when the portfolio is more concentrated. The intuition is that the estimation error of the betas averages out more in diversified portfolios.

---

**12.4 Factor-Mimicking Portfolios of Time Series**

A problem related to hedging a portfolio using a tradeable security is that of finding a portfolio that is close to a nontradable security. Such time series abound in practice. A quantitative portfolio manager may be interested in trading them for a few reasons. First, the time series may show high correlation to the securities in her investment universe and therefore the time series could serve as a useful hedging instrument. A common use case is that of the macroeconomic update time series, which has some market impact but is not directly tradable. Developing a tradeable portfolio that “tracks” the time series has no value for her. Lastly, just verifying how well we can track a time series is interesting in itself. It shows us whether the time series is of concrete use. The occasional analysis that finds non-tradeable themes are full of sound and fury, usually signifying sell-side research fees.

We introduced the ingredients of our problem earlier in the chapter. We have `K` assets with returns `r_h`, and a non-tradeable time series `r_c`. We keep the original notation. Since we would not be satisfied with implied factor exposures to this complex form, we have the following ingredients:
*   Two periods and one realized return. Investment decisions are made in period one, profits are realized in period two.
*   `K` assets with returns `r_h`, with covariance matrix `Ω_h`.
*   Nontradable time series `r_c` with returns `r_c`, with covariance matrix `Ω_c`.
*   Returns `r_c` and `r_h` are jointly distributed with covariance matrix `Ω = diag(Ω_c, Ω_h, ..., Ω_h)`.

The problem asks to minimize the tracking error between the time series and a portfolio.
`min_w E[ (r_c - wᵀr_h)² ]`
We condition on `r_h`, as we did earlier in this chapter:
`E[r_c|r_h] = E[r_c] + Ω_{ch}Ω_h⁻¹(r_h - E[r_h])`
`= μ_c + Ω_{ch}Ω_h⁻¹(r_h - μ_h)`
`= μ_c* + β*ᵀr_h`
`w* = (Ω_h + μ_h μ_hᵀ)⁻¹(Ω_{ch} + μ_c μ_hᵀ)`
`= (Ω_h + μ_h μ_hᵀ)⁻¹(Ω_{ch} + μ_c μ_hᵀ)`
The first-order condition on this unconstrained problem gives the optimal portfolio, which we transform by means of the Woodbury-Sherman-Morrison Lemma of the inverse matrix (see Equation A.7).
`w* = [Ω_h⁻¹ - (Ω_h⁻¹μ_h μ_hᵀΩ_h⁻¹) / (1 + μ_hᵀΩ_h⁻¹μ_h) ] (Ω_{ch} + μ_c μ_hᵀ)`
`= [Ω_h⁻¹ - (Ω_h⁻¹μ_h μ_hᵀΩ_h⁻¹) / (1 + μ_hᵀΩ_h⁻¹μ_h) ] Ω_{ch}`
`+ [Ω_h⁻¹ - (Ω_h⁻¹μ_h μ_hᵀΩ_h⁻¹) / (1 + μ_hᵀΩ_h⁻¹μ_h) ] μ_c μ_hᵀ`
Having done most of the heavy lifting, we close with a few remarks:
*   The beta estimation error `β*` serves as a regularizer for the covariance matrix. The larger the expected returns, the higher the importance of the regularization term.
*   When `μ_h = 0` (no estimation error), and `μ_c = 0` (zero return), the optimal portfolio is `w* = Ω_h⁻¹Ω_{ch}`, which is a scaling factor.
*   In a mean-variance portfolio, a minor point: it seems that the scaling factor is `σ_h²`, which would make no sense. The covariance matrix does contain `σ_h²`, though, so that dependency is effectively linear.
*   When `μ_h ≠ 0`, the optimal portfolio approaches, up to a constant, `Ω_h⁻¹μ_h`.
Once we have the optimal portfolio `w*`, hedging is straightforward, in the sense that we can employ Equation (12.6) to reduce the core portfolio’s risk.

**Exercise 12.5:**
Describe how you would hedge to a time-series factor (or an FMP of a time-series) on top of equity FMPs for a pre-existing model. (Hint: Orthogonalization.)

---

**12.5 Appendix**

**Proof**
[Proof of Exercise 12.2] We replace the decision variable `w_h = Px`. From the definition of `P`, it follows that `BᵀPx = x`, and
`xᵀPᵀΩ_h Px = xᵀ(BᵀΩ_h⁻¹B)⁻¹x`
It follows that the Optimization Problem (12.5) can be rewritten
`max_{x} α_cᵀ(w_c + Px) + μ_Fᵀ(Bᵀw_c + x) - (1/2)(w_c + Px)ᵀΩ(w_c + Px) + f(w_c, Px)`
`s.t. Bᵀ(w_c + Px) = x`
`σ_h² = σ_c² - xᵀ(BᵀΩ_h⁻¹B)⁻¹x + b_cᵀ(BᵀΩ_h⁻¹B)⁻¹x - xᵀ(BᵀΩ_h⁻¹B)⁻¹b_c + xᵀ(BᵀΩ_h⁻¹B)⁻¹x`
`w_c, x ∈ ℝ^n`
Assume that `μ_F = 0` and transaction costs equal 0. The objective function becomes
`α_cᵀw_c - (1/2)w_cᵀΩw_c + α_cᵀPx - (1/2)xᵀPᵀΩPx - w_cᵀΩPx`
`= α_cᵀw_c - (1/2)w_cᵀΩw_c + α_cᵀPx - (1/2)xᵀ(BᵀΩ_h⁻¹B)⁻¹x - b_cᵀ(BᵀΩ_h⁻¹B)⁻¹x`
which is minimized at
`x* = -(BᵀΩ_h⁻¹B)⁻¹BᵀΩ_h⁻¹b_c - (BᵀΩ_h⁻¹B)⁻¹BᵀΩ_h⁻¹α_c`
`= -[I + (BᵀΩ_h⁻¹B)⁻¹BᵀΩ_h⁻¹]b_c`

**The Takeaways**

1.  Hedging reduces portfolio risk by adding negatively correlated investments.
2.  Common hedging methods include market and currency hedging.
3.  Additional hedging techniques involve:
    *   Using FMPs from fundamental factor models.
    *   Hedging with futures or liquid assets capturing equity and non-equity risk (e.g., energy futures, sector ETFs).
    *   Employing thematic tradeable baskets offered by banks to hedge or speculate on specific risks.
4.  In a simple hedging scenario we have two assets: core (with expected return `μ_c`) and hedge (with expected return `μ_h = 0`) and want to maximize the Sharpe Ratio of the combined portfolio. The optimal hedge position is `x_h* = -β(r_c, r_h)x_c*`, with `β = ρ_ch σ_c / σ_h`.
5.  In practice, we remove the assumptions in simple hedging: accurate beta estimates, single trading instrument, negligible trading costs, hedging instrument with negligible expected return.
6.  In both cases, we will hedge less than the optimal hedge ratio of the simple hedging case.

**Notes**

1.  We are assuming, again, that the returns of the factors we want to hedge are zero, or negligible.
2.  Sometimes these are referred to as unspanned factors, because we receive no reward for holding their associated risk.
3.  Informally, if the number of assets is large, we should expect the variance of `w_cᵀη` to be small, so that the expected value is a good proxy for `w_cᵀη`.
4.  For a vector `x`, define `E_η[x]` as the average of the values `x_1, ..., x_N`.
5.  The Herfindahl index is usually defined for a set of `N` non-negative numbers `x_i` that sum to one: `H = Σ_{i=1}^N (x_i / Σ_{j=1}^N x_j)²`. It can be extended to arbitrary sets of numbers `x_i` by defining `H = (Σ_{i=1}^N |x_i|)² / (N Σ_{i=1}^N x_i²)` and applying the original definition.
6.  In Section 7.2.2, I defined `β* = (BᵀΩ_D⁻¹B)⁻¹BᵀΩ_D⁻¹r_m`. I have not been lucky enough to meet its members in the wild.

```

Okay, here is the Markdown compilation for Chapter 13.
     Everand

