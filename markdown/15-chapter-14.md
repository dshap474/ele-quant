**Chapter 14**
**Ex-Post Performance Attribution**

**The Questions**

1.  Is portfolio performance primarily due to skill or luck?
2.  What is the breakdown of PnL between factor-driven and idiosyncratic sources?
3.  In idiosyncratic space, is asset selection or sizing a more significant contributor to PnL?
4.  How can factor-based PnL be analyzed in a consistent and meaningful way?
5.  What are effective methods for decomposing portfolio performance in dynamic and complex markets?
6.  How does uncertainty in model parameters affect performance attribution accuracy?

“After the horses have run,” summarizes the plain sense of things.[1] So begins the most poetic, which describes well the spirit of this chapter. Out of metaphor, the “run” is the realized performance of our strategy, and the plain sense of things is our ability to understand what happened after the fact, namely:
*   Is our performance due to luck or skill?
*   How did we make or lose money? What is the contribution of factor PnL and idiosyncratic PnL?
*   In idiosyncratic space, did we drive our PnL? Asset selection or sizing?
*   Sizing is betting on the right side of a bet; the second is the ability to size appropriately asset bets that yield higher returns.
*   How can we explain factor PnL concisely and insightfully, i.e., using only factors that are of interest to us?

Performance attribution is about asking these questions. First, it provides the portfolio manager with a multi-faceted reality check. If she lost money, maybe she can explain the source of the loss, and identify countermeasures to apply going forward, sometimes the remedies are straightforward and contained in the output of the performance attribution itself. If she made money, maybe she did so as the result of unintended bets on factors that were not in her plan, and she can investigate.
The first principle is that you must not fool yourself—and you are the easiest person to fool.[2] This statement, made by Richard Feynman in his 1974 CalTech commencement address, holds true for scientists and traders alike. Second, performance attribution requires the information to be used as appropriately. The principal may be the hedge fund manager and the agent the portfolio manager, or, descending one step down in the decision-making hierarchy, the principal may be the portfolio manager and the agent may be the analyst who works in the portfolio manager’s team. There are other benefits. A portfolio manager is bound to use a specific factor model for ex-ante portfolio construction. An ex-post limitation exists after the fact, though. We can look at the performance of the portfolio using a set of different risk factors. For example, a global risk model, sometimes too coarse for country-specific investing, could reveal cross-country exposures. We can also use statistical models in addition to fundamental models.

Performance attribution is conceptually simple but is not trivial. The remainder of this chapter is broadly organized into two parts. First, we introduce characteristics-based performance attribution (also known as holdings-based), and then review the concept of time-based attribution.

---

**14.1 Performance Attribution: The Basics**

Recall the short introduction to performance attribution in Section A.1.2. The PnL can be decomposed into the sum of factor and idiosyncratic components. The performance decomposition process is slightly more involved. Trading time is not discrete, whereas performance attribution occurs in discrete time. To reconcile the two views, the time axis is partitioned into intervals delimited by epochs `t_k`. Denote the PnL by `PnL_k`, the PnL driver `(r_k - r_{f,k})`, and the total returns `r_k`, and the total factor returns `f_k`. In the case of discrete returns, we define the net holdings `w_k`. We have done previously `b_k = Bᵀw_k`. Then we can isolate the trading PnL with the decomposition
`PnL = Σ (PnL_k - r_{f,k}w_k) + r_{f,k}w_k`
`= Σ (PnL_k - r_{f,k}w_k) + Σ b_kᵀf_k + Σ ε_kᵀw_k`
(trading PnL) (factor PnL) (idiosyncratic PnL)
(position PnL)

The sum of factor and idiosyncratic PnL is sometimes referred to as position PnL. This is the PnL we would experience if we traded instantaneously and with no transaction costs, so that it is very far away from the actual PnL, even at the terminus of the interval’s total returns. To fix ideas on the interpretation of the trading PnL, it is helpful to consider the case of an idealized high-frequency trader (HFT). Let the epochs be the close of trading days. The HFT ends the day flat: `W_N = 0`. The sum over `k` of the trading PnL is the total PnL. It is also the PnL from providing “intraday alpha, i.e., “price discovery,” compensation for providing liquidity by submitting limit orders and removing a fraction of the bid-ask spread, and costs incurred by taking liquidity by submitting market orders.

The factor PnL can be decomposed into separate time series for the contribution of each factor:
`(14.1) Factor PnL = Σ_{j=1}^m [ Σ_{t=1}^T b_{j,t} f_{j,t} ]`

This could be the end of a simple story: take portfolio snapshots at each epoch, decompose PnL into three terms, and then dive into the contribution of individual factors and of individual securities to idiosyncratic PnL. But the world is more complex. First, we need to unveil the illusion of certainty that comes with the simple decomposition of Equation (14.1).

---

**14.2 Performance Attribution with Errors**

**14.2.1 Two Paradoxes**

To motivate the importance of having a more nuanced view of factor-based performance attribution, we introduce two paradoxical facts, both related to FMPs:
*   Factor-Mimicking Portfolios have idiosyncratic risk but not PnL. Each FMP has, by construction, zero exposure to all other factors, and therefore `b_kᵀf_k = 0`. Necessarily, the FMP has non-zero idiosyncratic variance `σ_ε²`. Otherwise, the FMP would be an exact linear combination of the factors, which can be seen intuitively by the fact that the return of the factor is the return of the portfolio itself. More rigorously, let `P` be the matrix whose columns are the FMPs, as defined in Equation (7.2). Then their idio PnL is
    `Pᵀε_k = (B(BᵀΩ⁻¹B)⁻¹)ᵀ(I_n - B(BᵀΩ⁻¹B)⁻¹Bᵀ)Ωε_k = (B(BᵀΩ⁻¹B)⁻¹)ᵀ(Ω - B(BᵀΩ⁻¹B)⁻¹BᵀΩ)ε_k = 0`
    and therefore the idio PnL is null. This holds for all factor portfolios, including those that have an idiosyncratic variance percentage close to 99%, and for all periods. This is especially concerning given that factor model performance is often evaluated via factor portfolios.
*   Factor-Neutral Portfolios. On the other side, consider a portfolio `w` with no exposure to any factor: `Bᵀw = 0`. Hence, its variance comes entirely from idiosyncratic risk: `wᵀΩw = wᵀεεᵀw`. Consider a portfolio `w = λv`, where `v` is an FMP and `λ ∈ ℝ`. The idiosyncratic risk of this portfolio is the same for any value of `λ`, since `v` has no idio PnL. However, the idiosyncratic volatility of `w = λv` depends on `λ`: `σ_ε(λv) = λ²σ_ε(v)²`.
    Hence, even though the same sequence of returns `r_k` generated via simulation of portfolio `w` with different volatility `σ_ε(λv)`, we can make the realized idiosyncratic PnL of the portfolio arbitrarily different than the predicted idiosyncratic volatility, thus greatly undermining the credibility of the model. How can this be?
One could argue that in practice factor portfolios do have low idiosyncratic PnL. This is due precisely to the construction of the portfolio, so that factor portfolios as of time `t` are slightly stale when applied to time `t+1`. This criticism doesn’t address the concerns exemplified by the paradoxes for two reasons. First, because even in the ideal case in which the world is stationary and we have accurately estimated parameters, the average factor PnL of a factor-neutral portfolio would be in any event much smaller than what would be compatible with the idiosyncratic volatility predicted by the model.

(Right Sidebar Text from Page 590)
In the next three sections I present a possible solution to these paradoxes. The overall takeaway is the analysis is that the returns of the FMPs are estimates of the actual factor returns, and not the factor returns per se. In addition, the portfolio’s factor exposures `b_k` are also estimates. Accordingly, both factor PnL and idiosyncratic PnL have distributions in addition to their true factor returns. Once we account rigorously for the estimation error, the factor PnL and idiosyncratic PnL can be characterized as random variables whose mean and several moments can be estimated from the model and portfolio. The next section presents the details about model estimation. The next section derives the main formulas. We then give explanations for the paradoxes.

**14.2.2 Estimating Attribution Errors**

Let us rewrite the attribution equations, but paying attention to the fact that we are using factor and idiosyncratic return estimates `f̃_k, ε̃_k`. We consider the case of a time-independent factor model. Recall from Section A.1.1 that the factor returns can be written as
`f̃_k = f_k + η_k, η_k ~ N(0, B(BᵀΩ⁻¹B)⁻¹)`
Analogously, for the idiosyncratic returns, we have
`ε̃_k = r_k - B f̃_k`
`= r_k - B f_k - B η_k`
`ε̃_k ~ N(0, B(BᵀΩ⁻¹B)⁻¹Bᵀ)`
`(estimated factor PnL)_k = w_kᵀB f̃_k`
`(estimated idiosyncratic PnL)_k = w_kᵀε̃_k`
`= w_kᵀ(r_k - B f̃_k)`
`(true factor PnL)_k ~ N(0, (w_kᵀB)(BᵀΩ⁻¹B)⁻¹(Bᵀw_k))`
When we attribute the PnL over multiple periods, we have
`(true factor PnL)_T = (estimated factor PnL)_T = Σ_k w_kᵀB f̃_k`
`(true idiosyncratic PnL)_T = (estimated idiosyncratic PnL)_T = Σ_k w_kᵀε̃_k`
Finally, this gives us two useful results. First, it provides confidence intervals around the attributed PnL:
`(true factor PnL)_T ~ N(Σ_k w_kᵀB f̃_k, Σ_k (w_kᵀB)(BᵀΩ⁻¹B)⁻¹(Bᵀw_k))`
`(true idiosyncratic PnL)_T ~ N(Σ_k w_kᵀε̃_k, Σ_k w_kᵀB(BᵀΩ⁻¹B)⁻¹Bᵀw_k)`
The idiosyncratic PnL is no longer independent of the hedge `Aw_k`. A portfolio with large idiosyncratic PnL must therefore have large factor PnL. The uncertainty is linear in hedge.

If, for example, we observe a negative idiosyncratic PnL over a given time interval, we can determine whether it falls inside the 95% confidence interval or not. The same applies to factor PnL. Overall, this leads to the true factor, factor and idiosyncratic PnL to be in general negatively correlated. Take the case of a constant portfolio, and constant factor exposures `b_k`. The covariance between factor and idiosyncratic PnL is given by
`-b_kᵀ(BᵀΩ⁻¹B)⁻¹b_k`
This is sometimes observed in practice.

**14.2.3 Paradox Resolution**
We first discuss the paradoxes introduced in the first section.
*   Factor Portfolios: Factor portfolio `j` has exposure vector `b_j = (0, ..., 0, 1, 0, ..., 0)`
    ...where the 1 is in the `j`-th position, so `(b_j)_k = δ_{j,k}`. Therefore
    `(true factor PnL)_j ~ N(f_{j,T}, T(BᵀΩ⁻¹B)⁻¹_{j,j})`
    So the factor portfolio has a non-zero mean idiosyncratic PnL, whose variance grows linearly in `T`.
*   Factor-Neutral Portfolios: Let `w` be a portfolio with no exposure to any factor, i.e., `Bᵀw = 0`. The portfolio `w = λv`, where `v` is the first FMP, has exposure
    `b_v = (λ, 0, ..., 0)`
    The factor and idiosyncratic PnL are
    `(true factor PnL)_T ~ N(λ Σ_t f_{1,t}, λ²T(BᵀΩ⁻¹B)⁻¹_{1,1})`
    `(true idiosyncratic PnL)_T ~ N(0, λ²T(BᵀΩ⁻¹B)⁻¹_{1,1})`

(Right Sidebar Text from Page 594)
**Insight 14.1: Reporting standard errors for attributions**
When reporting factor-based performance attributions, always include (either graphically, or in tabular form), the standard errors of the factor and idiosyncratic PnLs, using the formulas from Section 14.2.2 (or (14.2)). This will help the portfolio manager better understand the uncertainty associated with her attributed performance.

Summing up, the current factor-based attribution methodology universally assigns a numeric factor and idiosyncratic PnL to a strategy; these are deterministic functions of the portfolio over time, the stock returns, and additional available data, such as asset characteristics. Ignoring the estimation error of these attributions leads to misleading conclusions, and does not help in reducing uncertainty in performance of factor portfolios and hedged portfolios, or in the portfolio risk management and to understanding the performance of a strategy. As a simple resolution to these paradoxes, we saw that, even if we are employing the true factor model, the returns of the FMPs are unbiased estimates of the actual factor returns, and not the factor returns themselves. Given this, the estimation error can propagate its impact to the performance attribution process, and view the factor and idiosyncratic PnLs as random variables for which we have the full distributions (under the assumption of normality of returns) and the confidence intervals.

---

**14.3 Maximal Performance Attribution**

A different way to summarize the previous section is: do performance attribution results make sense? The common sense answer is “yes.” In this section, we go deeper into this question by covering exotics, a similarly ex-ante summary, do performance attribution, but to reduce confusion. If we had to attempt a parallel to real life, performance attribution is like falling in love: fundamentally good, but certainly dangerous, and potentially confusing. Where does the confusion come from? Consider the case of a portfolio manager who has generated a large positive PnL from a large exposure to a factor, say, momentum. He then cuts momentum exposure to zero, as a defensive measure. The day after, the factor has a very large negative return. We ask: is the portfolio’s reported PnL equal to zero? The answer is no. Another way to state the issue is that the relationship between ex-ante performance attribution and ex-post performance attribution is not one-to-one. A portfolio’s ex-ante performance is given by the covariance between portfolio returns and the factor’s returns, divided by the factor’s variance. In formulas:
`β(w, f_k) = b_kᵀΩ_{f_k} e_k / (e_kᵀΩ_{f_k} e_k)`
The beta is in general non-zero even if `b_k = 0`. Because
`β = Σ_{j≠k} ρ_{kj} b_j (σ_k / σ_j) + σ_k`
Factors other than momentum, but that are correlated to it, are responsible for the transmission of the shock.

Let us go through another example. You are developing a risk model with a country factor (whose loadings are all ones) and a historical beta factor. You have the option of a missing data model and beta loadings. The choice does not affect individual asset PnL, and aggregated factor PnL does not change if you score or not. Nor is present, and aggregated factor risk does not change if you score or not the individual beta and country factors change. The puzzling attribution in the beta, does much less so in the “right” factor model. What can we say? This is an extreme example of the fact that the factor PnL can have factor drawdowns. It is possible that the PnL be spread across multiple factors and that no factor stands out. It is also possible that these factor losses may be correlated. For example, losses in momentum could be “explained” as momentum losses, or crowded losses, or value losses, or even sector losses, with positive weights. The PnL associated with a factor depends on the representation of the factor itself in the risk model. By this, we mean that the information contained in a given set of factors can be represented in different ways. The same factor may have zero correlation to other factors in one representation, and positive correlation in another. We can even choose to have a single factor that explains the most variance. Performance attribution and risk to a subset of factors, such that it explains the PnL and the risk of the portfolio as much as possible?

The answer is in the affirmative. There is a procedure to assign unequivocally maximum risk and PnL to a subset of factors. There are four different ways to formulate and model the problem, all yielding the same result.
We introduce some notation. Denote the sets
`U := {1, ..., m}`
`S := {1, ..., p}`
`S̄ := {p+1, ..., m}`
so we write `f_S` instead of `f_U` and `Ω_{f,S}` instead of `Ω_{f,U}`.
1.  Maximal Cross-Sectional Factor Explanation. Consider the problem of describing the asset returns as a function of the returns of factors `S`, as well as possible, i.e.,
    `r = B_S f_S + B_S̄ f_S̄ + η`
    where `β ∈ ℝ^{n×p}` and `η` is uncorrelated with `f_S`. By construction, this is the maximum amount of returns we can attribute to factors `S`. Once we identify beta, the return attributed to factors `S`, `f_S*`, which is in general different than `B_S f_S`. We solve the problem
    `(14.3) min_{B_S, B_S̄} E[||r - B_S f_S - B_S̄ f_S̄ - η||²]`
    `s.t. r = B_S f_S + B_S̄ f_S̄ + η`
    `B_S ∈ ℝ^{n×p}, B_S̄ ∈ ℝ^{n×(m-p)}`
    This is equivalent to
    `min_{B_S, B_S̄} E[||f_S - B_S f_S||²]`
    which is solved by `B_S = I_p`. Then the attribution using factor set `S` is given by:
    `(14.4) wᵀB_S f_S = wᵀ[β_S, 0] [f_S; f_S̄]`
    The term `wᵀB_S f_S` is the maximal attribution to factors in `S`. When factors in `S` are uncorrelated to factors in `S̄`, the factor covariance matrix `Ω_f` is block-diagonal and `B_S = β_S`. The maximal attribution is the same. But in general, the factors in `S` and `S̄` are correlated and `Ω_f ≠ Ω_{f,S} ⊕ Ω_{f,S̄}`. Maximal attribution shifts the PnL attributable to factors in `S` from the other factors.
2.  Conditional Expectation. There is another way to interpret these formulas, based on conditional distribution of the multivariate Gaussian distribution. Given returns `r`, `f_S`, `f_S̄`, the conditional expected returns of factors in `S` are statistically and are given by the vector
    `E[f_S | f_S̄] = Ω_{f,S} Ω_{f,S̄}⁻¹ f_S̄`
    The formula for the maximal performance attribution is
    `b_SᵀE[f_S | f_S̄] = b_SᵀΩ_{f,S} Ω_{f,S̄}⁻¹ f_S̄`
    and this is identical to the maximal attribution term in Equation (14.4).
3.  Maximal Portfolio PnL Explanation. Start with the factor PnL of the portfolio `w`, with factor exposure `b_S`. Try to explain as much of the PnL by means of the returns of factors in set `S`. In formulas, we solve the problem
    `min_{b_S^*} E[||b_Sᵀf_S - b_S^*ᵀf_S||²]`
    `s.t. b_S^* ∈ ℝ^p`
    and the PnL attribution is `b_S^*ᵀf_S`, which is, again, what we obtain in Equation (14.4).
    This suggests an interpretation of the vector `b_S^*` as the adjusted dollar bets of the portfolio to factors in set `S`.
4.  Uncorrelated Factor Rotation. We have seen in Section A.2 that factor models are not uniquely determined. One can transform the loadings matrix by right-multiplying it by a non-singular square matrix `C`, and correspondingly transform the factor returns by left-multiplying them by `C⁻¹`. The resulting risk model has factor covariance matrix `C⁻¹Ω_f(C⁻¹)ᵀ`. It makes the same predictions as the original risk model, and the total factor PnL attribution `wᵀB_S f_S` is identical, and the total factor PnL attribution `wᵀB_S̄ f_S̄` is not the same as `wᵀ[BC]_S [C⁻¹f]_S`. We ask whether there is an equivalent rotation that yields the above “maximal attribution” to factors `S`, and what is the PnL attributable to factors in `S̄` for this particular example, given the previous derivations. We need to find `C` such that
    `(14.5) [b_S^*, b_S̄^*] = [b_S, b_S̄]C⁻¹`
    Define the matrix `A_S = [I_{p,S}, 0]` and the rotation matrix `C` as
    `C := [[I_{p,S}, 0], [A, I_{S̄,S̄}]]`
    `C⁻¹ = [[I_{p,S}, 0], [-A, I_{S̄,S̄}]]`
    Direct calculation shows that
    `(14.6) [b_S^*, b_S̄^*]C = [b_S^*, b_S̄^*A + b_S̄^*]`
    which is the same as Equation (14.5). In the rotated risk model the covariance matrix is
    `(14.7) Ω_f(C⁻¹)ᵀ = [[Ω_{f,S}, Ω_{f,S}Aᵀ - Ω_{f,S S̄}], [AΩ_{f,S} - Ω_{f,S̄ S}, AΩ_{f,S}Aᵀ - AΩ_{f,S S̄} - Ω_{f,S̄ S}Aᵀ + Ω_{f,S̄ S̄}]]`
    The interpretation of the transformation is that it makes the first `p` factors independent from the remaining ones. The returns and volatilities of the first `p` factors are unchanged, and the volatilities of the remaining ones are reduced. This is unintuitive at first sight, but has a simple interpretation: we have orthogonalized the factors in `S̄` and pushed the explanatory power in the first `p` group. On the other side, the dollar exposures of a portfolio for the first `p` factors, on the other side, are changing from `b_S` to `b_S^*` (see Equation (14.5)), so the PnL and the probability of the performance attribution is increasing as well.

Let us go through an example. We have a factor strategy for which we run daily factor performance attribution, which is shown in Figure 14.1 (top). We select to maximalize the factors market, momentum, and crowding factors. After rotating the remaining factors, the performance attribution changes significantly and is shown in Figure 14.1 (bottom). The market is responsible for a higher loss, crowding factors are larger than in the regular attribution, whereas growth is responsible for a smaller loss.

**Figure 14.1:** Top: PnL from factor performance attribution. Bottom: Maximal attribution on three factors: market, momentum, and crowding.

We close this section with two observations. First, we focused on the “maximal contribution” factors. Alternatively, we could focus on the set `S̄`, i.e., the performance attributable to factors in `S̄` once we have removed “maximal attribution” factors. The model has been rotated so that the portfolio performance has been described by a smaller dimensional space.

Second, we can perform a nested maximal performance attribution. Instead of having a “maximal attribution” set and a “minimal attribution” set, we extend the approach to a partition of the factor set `{1, ..., m}` by factor sets `S_1, ..., S_k`. Factor set `S_1` explains the maximal PnL, and so on. The most granular instance is where `S_i = {i}`, so that we orthogonalize the model sequentially one factor at a time. In practice, however, it may be more sensible to create a coarse partition, every element of which describes a common theme. For example, we may have a “market factor” set comprising country, market, and volatility factors; then a “value factor” set comprised of earnings yield, earning variation, dividend yield, book-to-price, and quality; a “sentiment factor” set, an “industry” set, and so on. The steps involved in simple maximal attribution and nested attribution are described in Procedures 14.1 and 14.2.

**Procedure 14.1: Maximal attribution**
1.  Inputs: Factor covariance matrix `Ω ∈ ℝ^{m×m}`;
    Universe `U := {1, ..., m}`; set `S ⊂ U`; universe `S̄ := U \ S`;
    portfolio `w`.
2.  Set
    `B := bᵀw`
    `A := Ω_{f,S} Ω_{f,S̄}⁻¹`
    `C := [[I_{p,S}, 0], [A, I_{S̄,S̄}]]`
3.  Output:
    Factor maximal PnL: `PnL_S = b_SᵀA f_S̄`.
    for all `k ∈ S̄`, `PnL_k = (b_k - A_{k,S}b_S)f_k`.
    Rotated factor covariance matrix `Ω̃ = C⁻¹Ω_f(C⁻¹)ᵀ`.

**Procedure 14.2: Nested maximal attribution**
1.  Inputs: Factor covariance matrix `Ω ∈ ℝ^{m×m}`;
    `U := {1, ..., m}`; set partition `S_1, ..., S_p` of `U`;
    portfolio `w`.
2.  For `i = 1, ..., p`:
    `U_i := U \ (∪_{j=1}^{i-1} S_j)`
    Perform maximal attribution (Procedure 14.1)
    Set `S := S_i`, `Ω_f := Ω_{f,U_i}`, `S̄ := U_i \ S_i`.
    `B_i = B_i - B_i Ω_{f,S_i S̄_i} Ω_{f,S̄_i S̄_i}⁻¹`.
3.  Store `PnL_{S_i}` for `g = 1, ..., p` and the related risk model.
    `Σ⁽¹⁾   0    ...   0`
    `0    Σ⁽²⁾  ...   0`
    `...  ...  ...  ...`
    `0    ...  ...  Σ⁽ᵖ⁻¹⁾`
    `0    ...  ...   0    Ω`

---

**14.4 Selection versus Sizing Attribution**

In factor-based attribution, the idiosyncratic PnL of a strategy is the most crucial performance term, representing the PnL that cannot be explained by factor exposures. While factor-based attribution identifies the non-systematic portion of the PnL, it fails to explain the source of idiosyncratic performance. Portfolio managers often consider asset selection and sizing as the primary sources of their skill. Selection refers to the ability to be long on stocks with positive returns and short on those with negative returns. Sizing refers to the ability to size portfolio positions right when right than when wrong. These skills have practical implications for portfolio construction and can lead to improved risk-adjusted performance. Quantitative analysts have developed “tilting” and “hugging” metrics to quantify selection and sizing, respectively. These metrics typically involve calculating the percentage of holdings in the ratio between the average PnL of profitable and unprofitable investments. Despite their intuitive appeal, these measures have two drawbacks: they lack a direct relationship with profitability measures like the Information Ratio, and do not provide clear guidance for portfolio managers.

This section aims to address these problems. We show how a new selection-sizing decomposition achieves two objectives:
1.  It links through an analytical, interpretable formula the IR of a strategy to the selection, sizing, and breadth of a portfolio.
2.  It provides guidance for portfolio managers. In the case that the strategy has positive sizing skill and that it has negative sizing skill.
The IR is the expected value of the idiosyncratic PnL divided by its standard deviation. If we restrict our attention to a single period, an estimate of the IR is
`(Idio PnL)_t / (Idio Vol)_t`
An estimate for the IR that employs the available time series of portfolios in epochs `1, 2, ..., T` is
`IR̂ = (1/T) Σ_{t=1}^T (Idio PnL)_t / (Idio Vol)_t`
The IR can be expressed as a simple combination of intuitive terms. The decomposition is
`IR̂ = √(1/T) Σ_{t=1}^T (selection)_t × (diversification)_t + (sizing)_t`
The terms in the identity are:
A selection skill
`(selection)_t = (1/n) Σ_{i=1}^n ε̃_{i,t} sgn(w_{i,t})`
We score the idiosyncratic return `ε̃_{i,t}` of an asset to obtain `sgn(ε̃_{i,t})`. If `sgn(ε̃_{i,t})` and `sgn(w_{i,t})` have the same sign, then `sgn(ε̃_{i,t}) sgn(w_{i,t}) = 1`. Sizing of a security bet is a specific period and the contribution to selection is positive. The scoring puts assets with different volatility on the same scale, so that selection does not reward the magnitude of the return.
Diversification. Instead of worrying about the rational value of diversification, we use the dollar volatility of each position, defined as `w̃_{i,t} = |w_{i,t}|σ_{i,t}`. Then we define
`(diversification)_t = ||w̃||₁ / ||w̃||₂`
When all the dollar volatilities are identical, then the portfolio diversification is `√n`. If, on the other end, the portfolio is concentrated, then the portfolio diversification is 1. The diversification squared ranges between 1 and `n`, and can be interpreted as the effective number of assets. This diversification term has a well-known connection to the Herfindahl index, which is a measure of concentration. To be more specific, define weights
`x_i = |w̃_{i,t}| / Σ_j |w̃_{j,t}|`
The Herfindahl Index is defined as `H = Σ_i x_i²`. The relationship is then
`(diversification)_t² = 1/√H`
The relationship between diversification and portfolio construction was first explored by Bouchaud et al. (1997).
The last term is sizing. It is equal to
`(sizing)_t = √(n) (1/n) Σ_{i=1}^n ε̃_{i,t} sgn(w_{i,t}) (w̃_{i,t} / ||w̃||₁)`
Here, `√(n)σ(ε̃, σ sgn(w))` where we treated the quantities associated to individual securities as empirical observations `ε̃_i`. The interpretation of sizing is that it measures the correlation between `sgn(ε̃_{i,t})` and `sgn(w_{i,t})` (the selection skill), and the dollar volatility `w̃_{i,t}`. Sizing is positive if, when the portfolio manager is right about the sign of a position, she is right about its size by having a relatively large position. In formulas, we first define
`(sizing)_t = (n / ||w||₁) Σ_{i=1}^n ε̃_{i,t} sgn(w_{i,t}) |w_{i,t}|`
(right side today) (bet size)
This equation can be used in several ways. To achieve a higher IR, a portfolio manager has the following three options:
*   Increase diversification. Markowitz famously said that diversification is the only free lunch in investing. This equation shows that benefits from diversification are via selection, i.e., the marginal benefit obtained by increasing diversification. This reasoning is not widely understood, however. Managers can increase diversification in two ways. The first one is by increasing the number of assets. This does not require additional effort. Alternatively, the portfolio manager could add stocks to the investment universe. This operation is not costless, since it would involve spending time on each stock and possibly commission costs. By increasing diversification, the manager may come to the stock selection from this decision.
*   Improve selection skill. The decomposition helps by providing a simple measure, which makes use of information dataset at a manager’s disposal. Once selection skill has been measured, and all the individual positions. Once selection skill has been measured, several actions are possible. For example, the portfolio manager can track the selection skill at the sub-industry or at the thematic level, or the portfolio manager can compare selection skill across various outside earnings.
*   Improve sizing skill. There is value already in having portfolio managers assess their sizing skill relative to selection; most portfolio managers overestimate their sizing skill, and find the true sizing skill, or even the overall trend, counterintuitive. If their sizing skill is negative, the portfolio manager should consider reducing the average position size. In doing so, they will eliminate the drag from negative sizing and magnify the benefit of stock selection, by maximizing breadth. If there is positive sizing skill, the portfolio manager can optimize the size of the high-conviction positions to maximize the IR. This is the subject of the next subsection.

**14.4.1 Connection to the Fundamental Law of Active Management**
The IR decomposition bears some resemblance to Grinold and Kahn’s Fundamental Law of Active Management (Grinold and Kahn, 1999). That law related the IR to the product of the Information Coefficient and of the breadth of the portfolio `√N`. This formula uses a different portfolio breadth—the effective breadth—which treats not all positions as equal. For example, a portfolio of 100 stocks with 99% of its gross exposure in each of two of them does not have the same breadth as a portfolio of 100 stocks with equal gross exposure in each stock and `N` in the remaining 99. In their seminal article, Bouchaud et al. (1997) present a modified mean-variance portfolio formulation that puts a lower bound on our definition of diversification. This results in using a shrunken covariance matrix (see also Ledoit and Wolf, 2003; DeMiguel et al., 2009b; Pedersen et al., 2021) and penalized covariance estimation methodologies (Ledoit and Wolf, 2003).

**14.4.2 Long-Short Performance Attribution**
The selection component of our performance attribution is linear, and therefore lends itself naturally to be further processed in different performance sub-terms. A natural way to decompose selection versus sizing is the sum of selection skill that arises from being on the right side of returns when positions are long versus when positions are short. The decomposition follows from the chain of equalities below:
`(selection)_t = (1/n) Σ_{i=1}^n ε̃_{i,t} sgn(w_{i,t})`
`= (n_{long}/n) (1/n_{long}) Σ_{i∈long} ε̃_{i,t} sgn(w_{i,t}) + (n_{short}/n) (1/n_{short}) Σ_{i∈short} ε̃_{i,t} sgn(w_{i,t})`
`= (n_{long}/n) (selection_{long})_t + (n_{short}/n) (selection_{short})_t`
where `n_{long}, n_{short}` are the number of long and short positions, and `Φ_{long}, Φ_{short}` are the fraction of the total portfolio positions that are long and short, respectively.
Summing up, in Figure 14.2 we show the dependency tree of the decomposition terms.

**Figure 14.2:** A taxonomy of performance attribution.
Total PnL -> Factor PnL, Idio PnL
Idio PnL -> Information Ratio
Information Ratio -> Diversification X (Long Positions), Diversification X (Short Positions), Sizing

---

**14.5 Appendix**

**14.5.1 Proof of the Selection versus Sizing Decomposition**

**Theorem 14.1:**
Consider a portfolio sequence `w_t ∈ ℝ^n`, and `n` names of iid idiosyncratic returns `ε̃_{i,t}`. Let `w̃_{i,t} = |w_{i,t}|σ_{i,t}` with `var(ε̃_{i,t}) = σ_i²`. Define the empirical Information Ratio:
`IR̂ = (1/T) Σ_{t=1}^T (Idio PnL)_t / (Idio Vol)_t`
Then the identity holds
`(14.11) IR̂ = √(1/T) Σ_{t=1}^T (selection)_t × (diversification)_t + (sizing)_t`
where the terms in the equation above are defined as follows:
`w̃_{i,t} := |w_{i,t}|σ_{i,t}`
`ũ_{i,t} := w̃_{i,t} / ||w̃||₁`
`ẽ_{i,t} := Ω⁻¹/²ε̃_{i,t}`
`(selection)_t := E[ẽ_{i,t} sgn(ũ_{i,t})]`
`(diversification)_t := √(n E[ũ_{i,t}²])`
`(sizing)_t := √(n) cov(ẽ_{i,t} sgn(ũ_{i,t}), |ũ_{i,t}|)`
Proof:
In period `t`, the risk-adjusted PnL of the portfolio at time `t` is given by
`IR̂_t = w_tᵀε̃_t / √(w_tᵀΩw_t)`
Set `w_t = Ω⁻¹/²w̃_t` and `ẽ_t = Ω⁻¹/²ε̃_t`. The vector `w̃_t` has a familiar interpretation. It is a portfolio whose positions are not expressed as NAV but rather as dollar volatilities of each asset. The return vector `ẽ_t` contains the idiosyncratic asset returns. Its covariance matrix is the identity. With these transformations, IR takes a simpler form:
`IR̂_t = w̃_tᵀẽ_t / ||w̃_t||`
This follows from the fact that the numerator is
`w_tᵀẽ_t = Σ_i w_{i,t} ẽ_{i,t} = Σ_i (σ_i w̃_{i,t}) (ε̃_{i,t}/σ_i)`
`= Σ_i w̃_{i,t} ε̃_{i,t}`
and the denominator is
`√(w_tᵀΩw_t) = √( (Ω⁻¹/²w̃_t)ᵀ Ω (Ω⁻¹/²w̃_t) ) = ||w̃_t||`
We can further simplify the formula by considering a breadth-rescaled percentage of the total dollar volatility
`IR̂_t = Σ_i w̃_{i,t} ẽ_{i,t} / ||w̃_t||`
`= Σ_i ẽ_{i,t} sgn(w̃_{i,t}) |w̃_{i,t}| / ||w̃_t||`
`= √(n) Σ_i ẽ_{i,t} sgn(w̃_{i,t}) (|w̃_{i,t}| / (√n ||w̃_t||))`
`= √(n) Σ_i ẽ_{i,t} sgn(ũ_{i,t}) |ũ_{i,t}|`
where we set
`ũ_{i,t} := w̃_{i,t} / (√n ||w̃_t||)`
We denote the cross-sectional empirical average and the cross-sectional empirical covariance
`E[X] := (1/n) Σ_i X_i`
`cov(X,Y) := E[XY] - E[X]E[Y]`
The formula becomes
`IR̂_t = √(n E[ẽ_{i,t} sgn(ũ_{i,t}) |ũ_{i,t}|])`
where we have used the notation `ũ` to denote the (double-indexed) product of two vectors, `ũ_i = u_{1,i} u_{2,i}`. Finally, in the last step we use the identity
`E[XY] = cov(X,Y) + E[X]E[Y]`
with `X = ẽ_{i,t} sgn(ũ_{i,t})` and `Y = |ũ_{i,t}|`. It follows that
`IR̂_t = √(n) (cov(ẽ_{i,t} sgn(ũ_{i,t}), |ũ_{i,t}|) + E[ẽ_{i,t} sgn(ũ_{i,t})] E[|ũ_{i,t}|])`
A possible interpretation of the above formula is as a sample of the realized IR over a single observation, or period. An estimate of the IR over the period `1, ..., T` is then given by its time series average:
`IR̂ = (1/T) Σ_{t=1}^T IR̂_t`
`= √(n) ( (1/T) Σ_{t=1}^T cov(ẽ_{i,t} sgn(ũ_{i,t}), |ũ_{i,t}|) + (1/T) Σ_{t=1}^T E[ẽ_{i,t} sgn(ũ_{i,t})] E[|ũ_{i,t}|] )`
This is equal to Equation (14.11) once we define
`(selection)_t = E[ẽ_{i,t} sgn(ũ_{i,t})]`
`(diversification)_t = √(n E[|ũ_{i,t}|])`
`(sizing)_t = √(n) cov(ẽ_{i,t} sgn(ũ_{i,t}), |ũ_{i,t}|)`

---

**The Takeaways**

*   Performance attribution helps determine if results are due to luck or skill and identifies sources of profit or loss.
*   Decomposing PnL into factor PnL and idiosyncratic PnL provides deeper insights into portfolio performance.
    Two paradoxes in performance attribution:
    1. Factor-Mimicking Portfolios: They have idiosyncratic risk but no idiosyncratic PnL, which is counterintuitive.
    2. Factor-Neutral Portfolios: They can have the same idiosyncratic PnL despite different volatilities.
    These paradoxes highlight the need to consider estimation errors in performance attribution models.
*   Accounting for estimation errors allows factor and idiosyncratic PnLs to be characterized as random variables.
*   Maximal Performance Attribution: Assigns as much PnL as possible to selected factors for clearer insights.
    Methods for maximal attribution include cross-sectional return explanation, conditional expectation, and model rotation.
*   Nested maximal attribution allows sequential attribution to different sets of factors, enhancing analysis granularity.
*   Selection versus sizing attribution decomposes idiosyncratic PnL into stock selection (right asset choice) and sizing (position size) components.
*   The Information Ratio (IR) links to selection skill, diversification, and sizing skill, providing a measure of performance.
*   Increasing diversification increases the marginal benefits of selection skill in a portfolio.
*   Enhancing selection and sizing skills can lead to better risk-adjusted returns and overall portfolio performance.

---

**Notes**

1.  W. Stevens, “The Plain Sense of Things,” in Stevens (1990).
2.  Non-idealized HFTs do not necessarily close the day flat, but instead rebalance the book and/or partially hedge it.
3.  By “any,” we mean that the return covariance matrix can be decomposed into the sum of a dense low-rank matrix and a sparse full-rank one.
4.  We use the notation `e_k` for the vector having a 1 in the `k`-th position and 0 elsewhere. (p. 596) We also assume that the factor portfolios have negligible idiosyncratic variance.
5.  We refer the reader to the Appendix, Section A.2.1 for derivations of the formulas below.
    A short derivation for two vectors `x,y ∈ ℝ^n`:
    `cov(X,Y) = (1/n) Σ x_i y_i - ( (1/n) Σ x_i ) ( (1/n) Σ y_i ) = E[XY] - E[X]E[Y]`
6.  The analysis presented here does not take into account transaction costs. This is a reasonable approximation for small portfolios. A more comprehensive model is possible, but outside of our scope.
   

Okay, here is the Markdown compilation for Chapter 15, the References, and the Index.

```markdown
Everand

