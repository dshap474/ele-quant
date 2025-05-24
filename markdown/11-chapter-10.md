**Chapter 10**
**Beyond Simple Mean Variance**

**The Questions**

1.  When does Mean-Variance Optimization (MVO) suggest shorting an asset with positive returns, and even short selling with its counterintuitive implications, when justifiable?
2.  How do different types of constraints (e.g., on market exposure, trading costs, portfolio turnover) address investor preferences, regulatory requirements, or practical considerations in portfolio construction?
3.  What are the potential benefits and drawbacks of using penalties instead of hard constraints in portfolio optimization?
4.  How do forecast errors in expected returns affect the realized Sharpe Ratio of an optimal portfolio?
5.  In what way does the error in estimating the covariance affect Sharpe Ratio?

---

**10.1 Shortcomings of Naive MVO**

Before considering more complex optimization problems, let’s work through a simple example—perhaps the simplest instance of the simplest optimization problem—to illustrate the implications of MVO. We have just two assets, with non-negative Sharpe Ratios $s_1, s_2$. Their returns have correlation $\rho$. The inverse of the covariance matrix is
`C⁻¹ = 1 / (1 - ρ²) * [[1, -ρ], [-ρ, 1]]`
So from Equation (9.16), the optimal volatility allocation is
`w*ᵢ = κ / (1 - ρ²) * (sᵢ - ρsⱼ)`
where $\kappa > 0$ is a parameter determined by the risk tolerance or by the constraint on total variance (see Section 9.2.4). If $\rho > 0$, we have three cases. Consider first the case where $s_1 > \rho s_2$ and $s_2 > \rho s_1$. In this case, weights are positive, and asset 2 acts as hedge. Shorting it is beneficial because (a) it has no cost (zero expected return), (b) it reduces the volatility of the portfolio, since it is positively correlated to asset 1. When the Sharpe Ratio of asset 2 is positive, then there is a cost to shorting it. If the asset is “too good” to short, shorting it no longer improves the Sharpe Ratio; the correlation must exceed the threshold $s_2/s_1$.

Even though the recommendation to short an asset with positive return is explainable, it is probably at odds with the intuition of many readers. If two assets are very correlated, wouldn’t it be preferable to go long both, thus averaging out the signal error? We will make this reasoning more rigorous by assessing the impact of estimation error on expected returns and on the correlation.

Impact of errors in forecasted Sharpe Ratios. We denote the true Sharpe Ratios $s_1, s_2$ and assume that the error between true and forecasted Sharpe Ratios is bounded by `||s̃ - s|| ≤ ε`. We choose optimal dollar volatility based on $s_1, s_2$ and true Sharpe Ratios $s_1, s_2$, and derive performance that is a function of $s_1, s_2$. The realized expected return is
`E[PnL] = κ / (1 - ρ²) * (s₁² - ρs₁s₂ + s₂² - ρs₁s₂)`

(Right Sidebar Text from Page 440)
**Insight 10.1: A simple linear-quadratic problem**
Let $a, x_0 \in \mathbb{R}^n$. The problem
`min_{x} (a, x) | ||x - x₀||² ≤ ε²`
has solution
`x* = x₀ - a / ||a|| ε`
`(a, x*) = (a, x₀) - ||a|| ε`
In the worst case, we solve the problem
`min E[PnL]`
`s.t. ||s̃ - s|| ≤ ε`
`s̃ ∈ ℝ²`
We leave the solution as an exercise (also, see Insight 10.2); the relative reduction in PnL is
`√((s₁ - ρs₂)² + (s₂ - ρs₁)² ) / (s₁² - 2ρs₁s₂ + s₂²) * ε`
This is the relative loss in Sharpe, since the volatility of the portfolio is unaffected by return forecast error. Figure 10.1 shows numerical results for two assets, assuming an error $\epsilon = 0.1$ in Sharpe Ratio estimation. As $\rho \to 1$, the perhaps counterintuitive, actual difference in forecasted versus realized Sharpe Ratios are higher. Notice that high correlation makes things worse. In all scenarios, the percentage in efficiency is significant. It is of course lower for higher Sharpes because the relative forecasting error is smaller, and is higher for higher correlations. In all cases it amounts to 10% and can be as high as 50%.

---
(Continuation from Page 444)
that the estimation error is bounded `||ρ̃ - ρ|| ≤ ε`. The error in estimated correlation affects the volatility; the return is not affected. The Sharpe Ratio is minimized when the realized volatility is maximized.
`max (v')ᵀ C̃V'`
`s.t. ||ρ̃ - ρ|| ≤ ε`
`ρ̃ ∈ ℝ`
In this case the worst-case realized relative volatility (exercise) is
`√( (V')ᵀ C̃V' + 2ε|v'₁v'₂| )`
and the associated relative loss in Sharpe Ratio is
`√( (V')ᵀ C̃V' ) / √( (V')ᵀ C̃V' + 2ε|v'₁v'₂| ) - 1`
We show the impact of the error in Figure 10.2 for a reasonable error in correlation estimate of $\epsilon = 0.1$. Moreover, in periods of crisis, the error can be larger (albeit not systematically so). Figure 10.2 shows the impact of correlation error on Sharpe.

**Figure 10.1:** Level plots of the loss of PnL (and Sharpe Ratio) as a function of the Sharpe Ratio of two assets, assuming a maximum error ε in the Sharpe Ratio norm. Parameters: ε = 0.1; Correlation ρ = 0.1 (top), ρ = 0.5 (bottom).

**Figure 10.2:** Fraction loss in Sharpe ratio for two strategies with Sharpe Ratios of 1 and 2, a return correlation ρ = 0.3, and error ε ranging from 0 to 0.3.

**Insight 10.2: Degradation in performance due to forecasting error**
When we use naive MVO optimization, the degradation in Sharpe Ratio arising from forecasted (ex-ante) parameters for volatilities and returns versus realized values (ex-post) can easily range in the 10-50%.

---

**10.2 Constraints and Modified Objectives**

Equation (9.16) is the starting point for more complex optimization problems. This section details some of the extensions: short-term constraints, long-term constraints, and implementation considerations. In applications, optimization formulations differ widely because they address a wide range of concerns:
*   Investor’s preferences: “Keep medium-term momentum exposure exactly equal to zero.”
*   Tactical considerations: “Don’t trade this stock because it could be acquired tomorrow” or “short this stock because it could be acquired tomorrow,” both of which represent constraints.
*   Regulatory considerations: “The portfolio must be long only.”
*   Fiduciary considerations: “The returns must track a benchmark, i.e., the difference in returns between the portfolio’s returns and the benchmark’s returns must not exceed a certain tracking volatility.”
*   Implementation considerations: “The objective function must include the trading costs.”

From a modeling viewpoint, constraints can take several forms. We introduce these first, and then we map them to the applications at hand. The “mapping” part will be brief, since it can never have been implemented, or is normally being done by an asset manager on a case-by-case basis. As in the previous chapter, we will focus on the linear group, and communicate with the latter.

**10.2.1 Types of Constraints**

Although one can imagine infinite types of constraints, some of them are much more common than others. We review them below.

Linear constraints. These can be inequality or equality constraints:
`Aᵀw = c` (Equality constraints)
`Aᵀw ≤ c` (Inequality constraints)
These are perhaps the most common constraints in financial optimization. For example, some strategies are required to be long-only. The constraint is simply
`w ≥ 0` (Long-Only constraint)
Extending this is a bound on maximum short and long size for a single position is only a small step. The main rationales for such constraints are many. There are natural limits due to maximum institutional ownership of a stock (say, no more than 5% of outstanding shares), or to maximum risk concentration in a stock (the idiosyncratic variance of a stock may not exceed a certain percentage of the total idiosyncratic variance, which translates to a linear constraint). Furthermore, we may impose a maximum liquidation cost requirement on all stocks, which also becomes a constraint on single-position sizes.

A slightly more complex constraint which does not seem linear at first sight, is on GMV. Recall that GMV is `Σᵢ |wᵢ|`. We can manage an upper bound on financial leverage that the fund can apply to its managed assets. The constraint can be turned into a linear one, by introducing ancillary variables representing the long and short side of a position, and additional constraints:
`x ≥ 0` (long positions)
`y ≥ 0` (short positions)
`w = x - y` (portfolio)
`Σᵢ (xᵢ + yᵢ) ≤ G` (GMV constraint)
A similar constraint is on the long versus short ratio, `LSR`. If we want the long/short ratio to be equal to a certain value, then the constraint is `Σᵢ xᵢ = K Σᵢ yᵢ`. This constraint is the same as the GMV constraint, with the exception of the factor `K`, which can be negative.
`Σᵢ xᵢ = K Σᵢ yᵢ` (Long/Short ratio constraint)
Yet another class of constraints is that on factor model exposures, and/or exposures to other asset characteristics not in the model. An example is the constraint on historical market betas `βᵢ`. The constraint then is `Σᵢ βᵢwᵢ = β₀`. The general form of factor exposure is verbatim that of Equation (10.3a).
A constraint on maximum portfolio turnover takes a similar form to the previous constraints that use absolute values. I am leaving it as an exercise to the reader. The turnover constraint may be either (poorly) justified to control costs, or by fiduciary requirements on portfolio turnover. A better way to model execution costs takes us in the domain of non-linear constraints.

Non-linear constraints. A constraint of a different nature is trading related. Trading costs are lumpy, and one approach has been to use piece-wise linear or piece-wise trading to account for asset-specific trading cost, in each portfolio rebalancing. This is equivalent to assuming linear transaction costs. We generalize this a little bit, and model trading costs as superlinear in the traded amount, or equivalently, as a quadratic cost of the form `Σᵢ cᵢ(wᵢ - w⁰ᵢ)² ≤ C`, where `cᵢ > 0` and `w⁰ᵢ` is the portfolio held at the beginning of the period. The constraint takes the form
`Σᵢ cᵢ|wᵢ - w⁰ᵢ| ≤ C` (Trading cost constraint)
where `w⁰ᵢ` is the portfolio held at the beginning of the period. The constraint is convex, so that the portfolio optimization problem has a unique solution.

Quadratic constraints appear naturally when we want to control risk at a finer resolution than that of total portfolio variance. For example, let `Bᵀw` be the principal subfactors in the factor covariance matrix, and let `D⁻¹` be the vector of style-factor exposures. Then a constraint on the maximum style-factor risk becomes
`(Bᵀw)ᵀD⁻¹(Bᵀw) ≤ ψ` (Style-factor vol constraint)
Risk constraints are often not only applied to the positions of a portfolio, but to the active positions of the portfolio itself. For example, consider a large-cap portfolio with a list of S&P 500 stocks, which is benchmarked to the S&P500 benchmark. The active holdings are `w_a = w - w_bench`. Tracking error is the volatility of the active portfolio, and is a measure of the freedom the portfolio manager has in selecting stocks. A constraint on the tracking error is
`(wᵀΩw)⁰.⁵ ≤ σ_max` (Tracking error constraint)
Non-convex constraints. Finally, there are a few constraint types that lead to a non-convex feasible region. Finding a global optimum is in general NP-hard. Convex solvers may either not accept such constraints, or may not converge. I would argue that, in most cases, these constraints should not be used on grounds of sensible modeling. I am presenting them both for completeness and as a cautionary tale.
The first constraint type is on the maximum number $N_{max}$ of assets in the portfolio. This is usually implemented by introducing 0/1 variables $x_i$, and by setting a maximum (large) absolute position size $M$. The constraint becomes
`|wᵢ| ≤ M xᵢ, i = 1, ..., n` (Max number of positions)
`Σᵢ xᵢ ≤ N_{max}`
`xᵢ ∈ {0,1} i = 1, ..., n`
The rationale for this constraint is that a very broad portfolio may be too burdensome to trade or manage. This combinatorial constraint can be handled by some commercial solvers for realistic problem instances. However, it makes the problem NP-hard. Usually it is much preferable to model trading costs directly, rather than include a constraint at all, or have a threshold for trading below which the trades of the optimal solution are set to zero. This usually has a negligible impact on optimality.

A very different type of constraint is on the minimum idio variance as a percentage of the total variance. We have mentioned this metric in Section 9.2.4. It is tempting to include a constraint of the form
`wᵀP_idioB_sB_sᵀP_idioᵀw ≥ ω₀ wᵀΩw`
or, equivalently,
`wᵀP_idioB_sB_sᵀP_idioᵀw - P_idioB_s(1 - P_idio)P_idioᵀw ≤ 0` (This seems to be a typo in the book, the formula is `wᵀ[P_idioB_sB_sᵀP_idioᵀ - ω₀Ω]w ≥ 0` or similar)
The problem is that the matrix `P_idioB_sB_sᵀP_idioᵀ - ω₀Ω` is in general not positive-definite, and therefore the constraint is not convex (exercise: prove it by providing an example).
A constraint type with a similar objective is to require a minimum idiosyncratic dollar volatility `σ_idio² ≥ σ_min²`. This is obviously a non-convex constraint, and its proponents should be excommunicated from the Orthodox Church of Optimization. A sensible approach is to simply upper bound the factor variance, or impose bounds of factor exposures, and test the impact of the bound on the portfolio’s performance.

Yet another excommunicable offense is imposing a lower bound on total volatility. I would not mention it, had I not witnessed actual humans proposing it.

In the same spirit, i.e., the goal of ensuring that the portfolio meets a minimum return, is a lower bound on GMV. The reason for these constraints is that the investor wants to ensure that, after accounting for return forecasts, trading costs, and risk constraints, the optimal portfolio is small, then maybe it should stay small. And if they really want to make it bigger (again, not advisable), one could loosen the upper bound on risk or underestimate the transaction costs.

---

**10.2.2 Do Constraints Improve or Worsen Performance?**

The naive answer to the title of this section is that—of course!—they worsen performance. If you reduce the feasible region of your optimization problem by adding a constraint, you will not get a better optimum. Specifically, if we maximize the Sharpe Ratio, adding constraints will degrade the Sharpe Ratio. This is true if the Sharpe Ratio, constraints, and expected returns are measured correctly. If we take estimation error into account, however, constraints may help. The next section interprets constraints as regularization terms for parameters entering in the optimization.

---

**10.2.3 Constraints as Penalties**

One alternative way to interpret a constraint in portfolio optimization is as a penalty term added to the objective function. Given a problem
`max f(x)`
`s.t. g(x) ≤ a`
with optimal solution $x^*(\lambda)$, there is a $\lambda^* > 0$ such that
`max f(x) - λ*(g(x) - a)`
has the same solution $x^*(\lambda)$. We used this result at the beginning of the chapter. The parameter $\lambda^*(\alpha)$ can also be interpreted as a sensitivity to the constraint’s right-hand-side parameter $a$. $\lambda$ is the marginal change in the optimum when we increase $a$ by $da$:
`df*(λ)/dλ = a(λ)`
Since a commercial solver returns both $x^*$ and $\lambda^*$, this means that we get sensitivity for free from the solution. This result also opens up a different interpretation. What if a potential constraint is not binding? Does this mean that the outcome, for the appropriate penalizing coefficient, is the same? Does this mean that the approaches are equivalent? The answer is no, and the remainder of this section is devoted to illustrating the difference.

First, let us focus our attention on the meaning of constraints and penalties. There are constraints that are communicable with the objective, and that are naturally expressed as penalties. For example, you could argue that the objective function should include trading costs and expected PnL in the objective function in the same unit (dollar) and it makes more sense to express the objective function as the difference of PnL and trading cost. The penalty parameter is simply one. What about constraints on risk? The total variance constraint has the dimension of dollar squared, and therefore our excommunicable way to add it to the objective is to add to the objective function is `λwᵀΩw`. This is possible in some optimization packages. However, if we know the approximate value of $\lambda$, of final volatility, we can choose a penalty parameter such that the adding a volatility term or a variance one gives a similar result. We do so by linearizing in the region of the optimum portfolio.
`-λ²(wᵀΩw - σ₀²) ≈ λσ₀²/2 - λ/2σ₀² λ²wᵀΩw`
The constant term is relevant to the optimization problem, and the volatility is loosely approximated by a variance.

A second class of constraints does not have an obvious interpretation. Should we add the constraint on GMV as a penalty? Or long-only constraints? The answer, somewhat surprisingly, is that adding those constraints as a penalty may actually help when we have an unconstrained portfolio, when the parameters in the model are not accurately estimated.

Let us start with an augmented version of Problem (9.16):
`max αᵀw - λwᵀΩw`
`s.t. ||w||² ≤ G`
whose penalized version is
`max αᵀw - λwᵀΩw - λ_G||w||²`
This problem can interpreted in many different ways. The first one is a simple rewriting of the quadratic term as
`λwᵀΩw + νwᵀ(I_n)w = wᵀ(λΩ + νI_n)w = wᵀΩ̃w`
The problem then is an MVO with a modified covariance matrix. The correlations $\rho_{i,j}$ of the original covariance matrix have been reduced by a factor
`(1 + ν/λ)⁻¹ ≈ 1 - ν/λ`
The asset variances have been increased, and covariances set closer to zero, each other, in the limit $\nu \to \infty$ they are identical. The norm constraint therefore has a “regularizing” effect on the solution. There are different optimization formulations that lead to the same solution of the Optimization Problem (10.10).

1.  Uncertain alpha (Goldfarb and Iyengar, 2003). Let us start with the assumption that the vector $\alpha$ is not known with accuracy. We make the assumption that the vector is distributed according to a multivariate Gaussian $\alpha \sim N(\alpha_0, \Sigma_\alpha)$. We still solve an MVO, using once more the certainty equivalence.
    `var*(wᵀα) = var(wᵀα*) + (w - w*)ᵀΩ(w - w*)`
    The MVO formulation is again the same as in that of Equation (9.16), but with a modified covariance matrix. As in the case of Equation (10.11), such a formulation will make more sophisticated calculations and shrink toward zero.

2.  Robust alpha (Pedersen et al., 2021). Instead of modeling alpha’s imperfect estimation by assuming that we know their distribution, we model their error deterministically, and adversarially: we know that the true alphas are within a certain distance $d$ from our estimate and, we are at the beginning of this chapter, we look at the worst case. The realized alpha is the worst possible one among the admissible realizations. In formulas, we solve
    `max_{w} α̃ᵀw - λwᵀΩ̃w`
    `s.t. α = argmin_{x̃} ||x̃ - x||² ≤ d²`
    We know the solution to the nested Problem (10.13a) from Insight 10.1; it is equal to `α̃ = α - d(w/||w||)`. Hence we solve
    `max_{w} αᵀw - λwᵀΩw - d||w||`
    This is similar, but not identical to, Equation (10.10): the norm penalty term is not squared. The same argument can be made to show that the norm and the norm squared are interchangeable, since the penalty constant $d$ is rescaled:
    `d||w|| ≈ (d/||w₀||) ||w||²`
    for $w$ close to $w_0 = w^*$ of the final solution.

3.  Robust factors (Ceria et al., 2012). We consider another instance of constrained optimization. A recurrent theme in this book is model misspecification. Factor models can be misspecified both in their factor structure and in their expected returns. Here, like in the case of robust alpha, we consider an unobserved factor, or a special case of misspecification. In effect, to worsen the Sharpe Ratio of the MVO portfolio. In order to reduce the impact, let us consider again an adversarial approach. Assume that there is a hidden factor, whose loadings we do not know, but whose volatility is given. We add this as a parameter that quantifies the impact of the missing factor.
    The new factor model contains an additional factor loading $v$ orthogonal to $B$. The covariance matrix is
    `Ω̃ = Ω + τ²vvᵀ`
    We solve
    `max_{w} αᵀw - λwᵀ(Ω + τ²vvᵀ)w`
    If $v$ is one of the basis vectors $e_i$, then the problem is
    `max_{w} αᵀw - λwᵀ(Ω + τ²/N Σᵢ eᵢeᵢᵀ)w` (The formula in book is `max αᵀw - λwᵀ(Ω̃ + τ²vvᵀ)w` then `max αᵀw - λwᵀ(Ω̃ + τ²/N Σᵢ eᵢeᵢᵀ)wwᵀ` which seems to have extra `wwᵀ`)
    So, yet again, we are solving an optimization problem with a penalized covariance matrix.

4.  Robust asset correlations (Boyd et al., 2017). Here we have another case of adversarial modeling that is expressed as a penalization term. Assume that we estimate the asset correlation matrix terms with some error independent of the asset pair, so that the difference between the estimated correlation and the true correlation is at most $d_{i,j}$. This is a common assumption in robust MVO problems and elsewhere (Natarajan’s “chooser” chooses a covariance matrix with the highest variance compatible with the error bound).
    `s.t. σ²(w) = arg max_{Δ} wᵀ(Ω̃ + Δ)w`
    `s.t. |Δᵢⱼ| ≤ dᵢⱼ, i,j = 1, ..., n`
    The objective of the nested problem is equivalent to
    `wᵀΔw = Σᵢⱼ wᵢΔᵢⱼwⱼ`
    Every term is maximized when
    `ρᵢⱼ = d × sgn(wᵢwⱼ)`
    and the objective function value is
    `(wᵀΔw)* = d Σᵢⱼ |wᵢwⱼ| Ω̃ᵢⱼ⁰.⁵`
    `= d (Σᵢ |wᵢ| Ω̃ᵢᵢ⁰.⁵)²`
    where `V` is a diagonal covariance matrix whose $i$-th diagonal term is the volatility of asset $i$. Let us plug this back in the original problem:
    `max αᵀw - λwᵀΩ̃w - λd||Vw||₁`
    And we have yet again a penalization term, which is, in this case, the square of an L1 norm of the portfolio weights. The function `||Vw||₁` is concave, so the optimization problem is convex. I summarize the penalization approaches in the table in Figure 10.3.

5.  Robust covariance matrix (Fabozzi and Wolf, 2009). Consider a different starting point to model robust covariance optimization. We assume that the adversary has a budget for the maximum cumulative squared error of the asset covariances `Σᵢⱼ Δᵢⱼ² ≤ d²`. This is the same as a bound on the Frobenius norm of the error matrix `||Δ||_F ≤ d`. The robust optimization problem is similar to the previous case:
    `max_{w} αᵀw - λwᵀΩ̃w - λσ²`
    `s.t. σ² = arg max_{Δ} wᵀ(Ω̃ + Δ)w ||Δ||_F ≤ d²`
    The strategy to solve this problem is similar to previous cases: the adversary maximizes a linear objective function with a norm constraint, see Appendix C.4 for the solution. In this case, `(wᵀΔw)* = d||wwᵀ||_F = d||w||²`, and the problem becomes an MVO with a quadratic penalization term.

**Exercise 10.1:**
Define the norm `||x||_A := ||A⁻¹x||`. Extend Goldfarb and Iyengar (2003), Reha Tütüncü and DeMiguel (2006) for additional interpretations of this penalty, and discuss their applicability to real-world settings.

| Approach             | Penalty                                                              | Parameter Interpretation                                                                                                |
|----------------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Uncertain Alpha      | λ_α times (sum squared times absolute value of α_hat)                  | λ_α times distance times volatility of missing factor                                                              |
| Robust Alpha (α̃)     | λ_α times (absolute value of hold) α̃                                   | λ_α times distance times volatility of hold-idio alpha_hat                                                              |
| Robust Factor        | λ_F times (sum squared times absolute value of hold) F                 | λ_F times distance times volatility of missing factor                                                              |
| Robust Correlations  | λ_ρ times (sum of abs value of the sub i comma j times hold) w squared | λ_ρ times distance times (sum of abs value of the sub i comma j times hold) w squared                                   |
| Robust Covariance    | λ_Ω times (sum of abs value of times hold) w squared                   | λ_Ω times distance times (sum of abs value of times hold cap omega hat sub hold) times omega hat sub hold times absolute value of... |

**Insight 10.3: Distinction between constraints and penalties**
Although they can yield the same optimal portfolio, the constrained and penalty version differ in two important ways. The first one is that the shadow price of the constraint is known only after the optimization. It is true that for some values the solution can be the same, but with a poor choice of the right-hand side of the constraint we don’t know the trade off between constraint limit and optimum value. This is not the case with a penalty: we set the price, and the price is often a more intuitive parameter than the quantity. For a risk-averse optimization, the price is unchanged, making comparisons easier. When the interpretation is clear, penalties are preferable. The second difference is in a continuous way: the first one is that the constrained solution may have no feasible solution, which is, in a loose sense, like saying that the price of the constraint is infinite. This is never the case with a penalized formulation, which is always feasible.

---

**10.3 How Does Estimation Error Affect the Sharpe Ratio?**

An investor starts with estimates of expected returns $\alphã$ and of the covariance matrix $\Omegã$. As derived in Section 9.2.4, the optimal portfolio is proportional to $\alphãᵀ\Omegã⁻¹$. The proportionality constant is irrelevant for the Sharpe Ratio. The realized Sharpe Ratio, however, is a function of the true expected returns and covariance matrix $\alpha, \Omega$:
`SR(α̃, Ω̃) = √(α̃ᵀΩ̃⁻¹α̃)`
We compare the realized Sharpe Ratio to the best Sharpe Ratio, based on the true values of $\alpha$ and $\Omega$, given by Equation (9.18):
`SR(α, Ω) = √(αᵀΩ⁻¹α)`
We call this the Sharpe Ratio Efficiency (SRE). It is important to study this quantity, because we want to know, at all times, whether we are losing a great deal of performance from inaccurate parameter estimation or large transaction costs. We will ask a few qualitative and quantitative questions, and see how far the analysis can take us.

First, prove an intuitive fact: incorrect estimates worsen performance.

**Theorem 10.1**
The Sharpe Ratio Efficiency is less or equal to one, and is equal to one if and only if $\Omegã⁻¹α̃$ and $\Omega⁻¹α$ are collinear.
Proof:
The SRE is
`SR(α, Ω̃) / SR(α, Ω) = (αᵀΩ̃⁻¹α̃) / (√(αᵀΩ⁻¹α) √(α̃ᵀΩ̃⁻¹α̃))`
Let
`a = Ω⁻¹/²α, b = Ω̃⁻¹/²α̃`
so that
`SR(α, Ω̃) / SR(α, Ω) = (aᵀb) / (||a|| ||b||)`
The Sharpe Ratio Efficiency is always less than one because of the Cauchy-Schwarz inequality, unless $\Omegã⁻¹/²α̃$ and $\Omega⁻¹/²α$ are collinear.

**10.3.1 The Impact of Alpha Error**

It is more useful to derive a lower bound on performance inefficiency, based on the estimation error of either expected returns or covariance.
We need to introduce a few basic results. Let the norm of a matrix be defined as the operator norm. Define the relative alpha error as
`||α̃ - α|| / ||α|| ≤ δ_alpha`
In the Appendix (Section 10.4.2) I prove the following result:
`SR(α̃, Ω_T) / SR(α, Ω_T) ≥ 1 - 2 ||Ω_T¹/²(α̃-α)|| / ||Ω_T¹/²α|| - ||Ω_T¹/²(α̃-α)||² / ||Ω_T¹/²α||²`
The alpha error inflates the Sharpe Ratio by magnified by the norm `||Ω_T¹/²||`. This is quite intuitive: the ratio of the largest eigenvalue to the smallest eigenvalue of the matrix `Ω_T`, or the ratio of the largest variance to the smallest variance associated to eigenfactors. The ratio is the condition number of a matrix. If we are operating in this span (all alphas are orthogonal to factor returns), then there is no variance misestimation, and the condition number is one. The ratio of the largest asset idiosyncratic variance to the smallest asset idiosyncratic variance.

**10.3.2 The Impact of Risk Error**

If there is $\kappa > 0$ such that
`||Ω̃¹/²Ω⁻¹/² - κI|| ≤ δ_risk`
then
`SR(α, Ω̃) / SR(α, Ω) ≥ 1 - 2δ_risk / (κ + δ_risk)`
This formula follows directly from Equation (10.26). At first sight, what is interesting about this result is how weak it is. Let us consider a few special cases. We define
`H := Ω̃⁻¹/²ΩΩ̃⁻¹/²`
1.  If the estimated covariance matrix is biased, but uniformly so, i.e., `Ω̃ = kΩ`, then `H = k⁻¹I` and there is no efficiency loss. We knew this already from the previous chapter. What happens in practice is that we would deploy a portfolio with the highest Sharpe Ratio, but incorrect volatility.
2.  Say that we correctly estimate the covariance matrix structure, only that we scale it. `Ω̃ = kΩ`. It can still happen that we have an SRE of one! This will happen if `α` is proportional to an eigenvector of `H` with a positive eigenvalue. Say the associated eigenvalue is `η²`. Then, use directly Equation (10.24):
    `SRE = (αᵀ(η²Ω)⁻¹α) / ||α||² = sgn(η)`
    Even more paradoxically though, the formula implies that if `α` is proportional to an eigenvector of `H` with negative eigenvalue, then the Sharpe Ratio Efficiency is $-1$. Incidentally, `H` is neither necessarily symmetric nor positive-definite, so a negative eigenvalue is indeed a possibility.
3.  But, you may argue, this is an exceptional circumstance. Consider a simpler but instructive case. We make the assumption that `Ω̃, Ω` have the same eigenvectors (aka eigenfactors) as `H`. In other words, they only differ because of the singular values.
    `Ω̃ = UΛ̃Uᵀ, Ω = UΛUᵀ`
    so that
    `H = Ω̃⁻¹/²ΩΩ̃⁻¹/² = UΛ̃⁻¹/²UᵀUΛUᵀUΛ̃⁻¹/²Uᵀ = UΛ̃⁻¹/²ΛΛ̃⁻¹/²Uᵀ`
    A great simplification. Decompose the eigenvalue ratio as $\nu_i = \lambda_i / \lambdã_i$. What is the lower bound on the SRE in this case for $\kappa$?
    `δ := min_i |(λ̃ᵢ - λ_i)/λ_i| = min_i |(νᵢ - κᵢ)/κᵢ|`
    and the optimal point is $\kappa^* = \nu_{avg} = n^{-1} \Sigma_i \nu_i$. We use these values in Equation (10.29) to obtain
    `SRE ≥ ( (1/2)(min_i ν_i + max_i ν_i) ) / √(min_i ν_i max_i ν_i)`
    `SRE ≥ 1 - (max_i ν_i - min_i ν_i)² / ( (min_i ν_i + max_i ν_i)² )`
    Hence the loss in efficiency arises from the fact that we misestimate the risk towards the volatility of eigenvectors of the error covariance matrix. If we underestimate them (or overestimate them) by the same constant, then we lose nothing, as noted in the first point above. Let us think of an adverse case. Say we estimate all volatilities exactly (`ν_i = 1`) except for one, which we underestimate by 50%. Then the worst-case loss of Sharpe Ratio can be 33%.

---

**10.4 Appendix**

**10.4.1 Theorem on Sharpe Efficiency Loss**

These theorems are informally introduced in Section 10.3.
We recall that
`||H⁻¹x|| ≤ ||H⁻¹|| ||x||`
and
`| ||Hx|| - ||Hy|| | ≤ ||H|| ||x-y||`
so that
`||Hx|| - ||Hy|| ≤ ||H(x-y)|| ≤ ||H|| ||x-y||`
Also, use the cosine rule:
`||a-b||²/||b||² = 2(1 - aᵀb / (||a|| ||b||) )` (assuming `||a|| = ||b||`)
`SR(α̃, Ω̃) / SR(α, Ω) = (ãᵀb̃) / (||ã|| ||b̃||)`
`= 1 - (1/2) ||ã-b̃||²/||b̃||²` (approximation for `ã ≈ b̃`)
where `a, b` are defined by Equations (10.23) and (10.28).

**Lemma 10.1:**
Let `H` be symmetric positive-definite, `x, y ∈ ℝⁿ`, and
`| ||x||/||y|| - xᵀy / (||x|| ||y||) | ≤ δ`
Then
`|| (x/||x||) - (y/||y||) || ≤ √(2δ)`
(Proof on page 475, right column)
Let $a, b \in \mathbb{R}^n$.
`||Hx|| / ||Hy|| ≤ 2 min{||H||, 1}` (This formula seems specific or misprinted)
The proof involves decomposing vectors and using eigenvalue properties, leading to:
`||Hx||/||Hy|| ≥ 1/√(1 + ||H||²δ²)`
This bound is tight, up to a constant. For example, consider the case of diagonal `H = diag(λ₁, ..., λ_n)`. We have
`x = e_n + δe_1, y = e_n`, with $e_i$ the standard basis vectors. We have
`||x||/||y|| - xᵀy/(||x|| ||y||) ≈ (3/2)ε := δ`
`||Ω̃⁻¹/²Ω¹/² - I_n|| ≤ δ`
Then
`SR(α, Ω̃) / SR(α, Ω) ≥ 1 - (1/2) ||Ω̃⁻¹/²Ω¹/² - I_n||_F² δ` (This formula seems to mix norms and parameters)
The image shows: `SR(α, Ω̃) / SR(α, Ω) ≥ 1 - 2 ||Ω̃¹/²Ω⁻¹/² - I_n|| δ`

**Theorem 10.2 (Misspecification of risk)**
If `||Ω̃⁻¹/²Ω¹/² - I_n|| ≤ δ`
then `SR(α, Ω̃) / SR(α, Ω) ≥ 1 - 2δ / (κ + δ)`
where $\kappa$ is the condition number of $\Omegã^{-1/2}\Omega^{1/2}$.

**Theorem 10.3 (Misspecification of alpha)**
If `|| (α̃/||α̃||) - (α/||α||) || ≤ δ_α`
then `SR(α̃, Ω) / SR(α, Ω) ≥ 1 - 2 ||Ω|| ||Ω⁻¹|| δ_α`
Proof:
Let `H_α := Ω̃¹/²Ω⁻¹/²`. Using this notation, the SRE Equation (10.24) and Condition (10.27) are:
`SR(α̃, Ω_T) / SR(α, Ω_T) = α̃ᵀH_α / (||α̃|| ||H_α||)`
`||H - H_α I_n|| ≤ δ`
Let `λ₁ ≥ λ₂ ≥ ... ≥ λ_n` be eigenvalues of `H_α`. The condition `||H - H_α I_n|| ≤ δ` is equivalent to `λ_i - κ*| ≤ δ` for all `i`. Then
`α̃ᵀHα / (||α̃|| ||Hα||) ≥ κ - δ`
`α̃ᵀHα / (||α̃|| ||Hα||) ≤ κ + δ`
`(α̃ᵀHα / ||α̃||²) / (αᵀHᵀHα / ||Hα||²) ≥ (κ-δ)/(κ+δ)`

---

**The Takeaways**

1.  Mean-Variance Optimization (MVO) can suggest shorting assets with positive expected returns if their correlation with other assets is high relative to their Sharpe Ratios.
2.  Errors in forecasted Sharpe Ratios and correlations can substantially reduce realized portfolio Sharpe Ratios, with parameter estimation errors worsening performance loss due to estimation error.
3.  Estimation errors in expected returns degrade portfolio performance, especially when errors are large relative to the accuracy of risk estimates.
4.  Constraining portfolio allocations (e.g., long-only, gross exposure limits) can reduce sensitivity to estimation error by limiting out-of-sample variation, but potentially lowering maximum achievable Sharpe Ratio.
5.  Non-linear and quadratic constraints, such as those on factor exposure or tracking error, allow for more tailored risk control in portfolio construction.
6.  Penalties can serve as a flexible alternative to constraints, especially when risk tolerance or trading costs need to be balanced within feasible limits.
7.  Robust optimization, which models uncertainty in alpha and covariance estimates, helps mitigate performance loss by introducing penalties that account for estimation risk.
8.  Shrinking or regularizing asset correlations and variances can improve realized Sharpe Ratios when covariances are uncertain, reducing sensitivity to estimation error.
9.  Transaction costs, market impact, and trading costs can reduce portfolio drift and control rebalancing expenses, particularly in markets with high transaction costs.
10. The augmented Lagrangian method often combines constraints and penalties to balance estimation error tolerance with implementation feasibility, allowing for adaptable portfolio construction.

---

**Notes**

1.  Before reinventing the wheel, know that some financial optimization packages abstract the modeling of the GMV constraint, so that you just have to specify it.
2.  ... for example, a few years ago, 130/30 portfolios were popular. These strategies managed net-long portfolios, with 30% of NAV invested in shorts and 130% in matching longs.
3.  Some solvers are able to understand from the description of the problem whether $f$ is convex or not.
4.  For example, see Clarke et al. (2022).
5.  J.P. Morgan (2007) finds that this analysis is not very large. See Jagannathan and Ma (2003) for an early contribution to the analysis of long-only constraints; the work by DeMiguel et al. (2009a) on trading penalties; Fan et al. (2012) on GMV constraints and Ceria et al. (2012); Saxena and Stubbs (2012) on penalties on the factor covariance matrix.
6.  A volatility constraint or penalty is in practice computationally more burdensome than a variance constraint or penalty.
7.  The third leg of the trading costs is market impact modeling cost. We will cover this in later chapters.
8.  Early papers on model estimation error, and the relative impact of alpha and estimation errors, are Michaud (1989), Chopra and W. Ziemba (1993), Broadie (1993).
9.  Let H be a symmetric positive-definite matrix and let X,Y be its SVG. Define H¹ᐟ²X = YΛ¹ᐟ²Vᵀ. Then ||H¹ᐟ²X - Y||_F² ≤ ||H¹ᐟ²X||_F² + ||Y||_F² - 2σ_min(H¹ᐟ²XYᵀ), which can be found in almost any linear algebra book. If x,y ∈ ℝⁿ, then |aᵀb| ≤ √aᵀa √bᵀb with the equality holding only if a = kb.
10. We use the notation e₁, ..., e_n for the standard basis in ℝⁿ.

```

Okay, here is the Markdown compilation for Chapter 11.
     Everand

