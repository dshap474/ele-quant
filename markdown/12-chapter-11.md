**Chapter 11**
**Market Impact-Aware Portfolio Management**

**The Questions**

1.  What are the main sources of trading costs, and how do they affect the profitability of a trading strategy?
2.  How can we model and quantify market impact, specifically temporary market impact, in the context of trading?
3.  How does market impact propagate over time, and what functional forms are commonly used to describe this propagation?
4.  How can finite-horizon optimization be structured to manage expected returns, market impact, and risk constraints over multiple periods?
5.  How does an infinite-horizon optimization framework differ from finite-horizon models, and when is it appropriate to use each approach?

Trading can possibly make money, but it surely costs money. When we execute a trade, we incur costs of all sorts: financing costs when we leverage our portfolio; borrowing costs when we short securities; commission costs to exchanges and other intermediaries. In addition to all of these costs, we pay for information, our or other market prices. Information that, once acted upon, that is, once we place a sell order, we somewhat push its price upward, which in turn makes the additional purchase of a security more expensive. What is less intuitive, however, is that this “price impact” can turn a strategy that is potentially profitable into a very unprofitable one. We see that there are two components to this cost. The first one is an instantaneous cost, the other governing price impact dynamics. The second is to use these laws to optimize the performance of our trading strategy. These two questions inform the organization of the chapter. In the next section, we provide a quick description of market impact models. Then we descend into the infernal circles of optimal execution.

---

**11.1 Market Impact**

A synthetic definition of market impact is the following (Kyle et al., 2020): “Market impact is the cumulative market-wide response of the arrival of new order flow.” The underlying process is complex and results from the joint contribution of several factors:

*   First, there is a direct reduction in inventory of the securities being bought or sold, causing a price movement.
*   Second, there is an informational effect: concluding a transaction may reveal private news about the future price of a security, as well as be indicative of future transactions (i.e., order flow).
*   Third, there is a mimetic effect: when participants, in the absence of information, imitate each other’s behavior, leading to temporary run-ups or run-downs.
*   Fourth, there is a strategic aspect. Even in the absence of information, strategic participants trade so as to exploit potential arbitrages arising from price impact.

Each one of these phenomena is complex, and hard to model. The total transaction cost associated to a trade is usually decomposed into three components:

Expected Transaction Cost = (spread cost)
                        + (temporary market impact)
                        + (permanent market impact)

The spread cost reflects the difference between the bid and the ask price. A market order “crosses” the spread, i.e., is executed at the best price offered by the contra-party. A limit order rests in the orderbook awaiting for execution, and thus consumes liquidity. On the other side, a limit order has an associated execution price, and adds liquidity. The average cost incurred in a trade is modeled as a function of the bid-ask spread, and assumed to be independent from other transactions. It is usually modeled as an asset-dependent percentage of the dollar traded.

The permanent impact is the price change that persists long after the order has been executed.

The temporary impact (also called “slippage”) is the price change that occurs during order execution and immediately afterwards, until price reaches equilibrium.

In the rest of the chapter, we focus on temporary impact, and effectively ignore both spread costs and permanent impact, although they are both interesting areas of research. The rationale for this choice is that both quantities are limited from the vantage point of a modeler specifically concerned with portfolio optimization, as opposed to one interested in understanding the mechanisms underlying price impact. Spread costs and temporary impact are by far the most important concerns because of their magnitude compared to permanent impact. Even our discussion of temporary impact is kept as short as possible, but not shorter.

**11.1.1 Temporary Market Impact**

We want to submit a large order, usually called a parent order or a meta-order. Depending on the asset class, we have different options for execution. For example, we could split the parent order into smaller “child” orders and execute them on exchange, or place them with a single algorithmic broker, whose mandate is to minimize cost, despite the broad array of trading venues and matching mechanisms, the expected market impact is described well by a common formula. Let `ẋ_t` denote the net number of shares traded up to time `t`, and with `ẋ_0` the trading rate, i.e., the number of shares traded per unit time. We may introduce two functions: the instantaneous impact function `f: ℝ → ℝ⁺`, and the propagator `G: ℝ⁺ → ℝ⁺`. We also have a positive constant term `κ`, which is a function of the security characteristics. The expected temporary market impact is given by the formula
`(11.1) E(P_T) - P_0 = κ ∫₀ᵀ f(ẋ(t))G(T-t)dt`
For an interpretation, consider the case of a marketable order: single-buy trade, short `N` shares, by selling at time `t_0`. Then the market impact `E(P_T) - P_0 = κ f(N)G(T-t_0)`. The market impact is proportional to the trade size, and is followed by a “relaxation” back to an equilibrium level. The function `G` is monotonically decreasing, thus reflecting the long-term convergence to an equilibrium level (see Figure 11.1). The function `f` should intuitively be monotonically increasing: the higher the trading rate, the higher the instantaneous impact. The overall impact (see Equation 11.1) is then a linear superposition of pulse trades, each one having an impact that relaxes back to equilibrium over time. The big question is then what is the functional form of `f` and `G`. Below are a few alternatives.

**Figure 11.1:** Market impact over time for a single trade executed at time `t₀`. The decay after `t₀` is proportional to `G`.

The functions are
`f(ẋ) = σ sgn(ẋ) |ẋ|^(β/2)`
`G(t) = δ(t)`
where `β ∈ (0.1, 1)` is the security’s volatility, `σ` is the total number of shares traded per unit time by all market participants, and `δ(.)` is the Dirac delta function.
We consider here and in the remainder of this book a buy order, i.e., `ẋ > 0`. The trading cost is
`C[ẋ] = κ ∫₀ᵀ dτ ẋ₀² ∫₀^(T-τ) dδ δ(t-s)`
`= κσ ∫₀ᵀ dτ ẋ₀^(β/2) ∫₀^(T-τ) dδ δ(t-s)`
`= κσ (Q/T)^(β)`
`= κσ (Q/T)^(β) T`
As an important example, we trade share quantity `Q` at constant rate in `[0,T]`, and the total traded volume in the market in the same period is `V = σT`. The ratio `Q/V` is usually referred to as the participation rate (or percentage of volume). From the formula above, the total unit costs are
`C = κσ (Q/V)^(β)`
The unit cost is decreasing in the execution time. By replacing `V = σT`, we have `C ∝ T^(1-β)`.

There is an argument made on the basis of physical dimensions; this suggests that `β = 1/2`. Toth et al. (2011) proposed an argument based on generalizing the Almgren-Chriss model. They assume that there are only three quantities that matter:
`Q`, which we interpret as the dollar value traded during a period.
`V`, which we interpret as the dollar value traded by all participants during the same period.
`σ`, the security’s volatility during the same period. Volatility has the physical unit of the inverse square root of time.
The transaction cost `c` is dimensionless and invariant in the units chosen for currency or time. If the argument is a polynomial of the input quantities above, we write
`c(Q, σ, V, T) = f(V^(γ)Q^(α))`
`(currency) / ( (currency)^(γ) * (time)^(γ/2) * (1/time)^(α/2) )`
from which `γ = -α` and `-γ/2 - α/2 = 0`. Set, without loss of generality `α = -1/2`. It follows that the cost is a function of
`V^(-1/2)Q^(1/2) σ`. `c(Q, σ, V) = f(V^(-1/2)Q)`
(Kyle, 1985; Huberman and Stanzl, 2004) This model is a special case of Almgren-Chriss’s model and precedes it historically. The functions are
`f(ẋ) = σ (ẋ/v)`
`G(t) = δ(t)`
The model is interesting in two regards. First, it is robust to price manipulation: a round-trip trade where an agent starts and ends flat, and is expected to extract a profit from the market impact they generate (Huberman and Stanzl, 2004; Gatheral, 2010). Second, it is empirically supported. We will use this model in Chapter 12, where we present a model of information trading.

(Obizhaeva and Wang, 2013) The functions are:
`f(x) = ẋ/v`
`G(t) = e^(-t/τ)`
The trading cost is
`C[ẋ] = κ ∫₀ᵀ dτ ẋ(τ) e^(-(T-τ)/τ) ∫₀^(T-τ) ds e^(-s/τ) ẋ(s)`
Consider again the constant-rate trade of `Q` shares over an interval `[0,T]`. The trading cost is:
`C = κ (Q/T)² τ [1 - τ/T (1 - e^(-T/τ))]`
`c = κ [τ/T - (τ/T)² (1 - e^(-T/τ))] (Q/V)`
Consider the two cases where timescales of execution and relaxation separate: “slow” execution `τ ≪ T` and “fast” execution `τ ≫ T`.
`c ≈ κ (Q/V) τ/T` if `τ ≪ T`
`c ≈ κ (Q/V) T/(2τ)` if `τ ≫ T`
For the slow execution case, the unit cost is inversely proportional to execution time, whereas it is independent of `T` when `τ ≪ T`. The overall market impact time series is shown in Figure 11.2. The Obizhaeva and Wang (OW) model (Obizhaeva and Wang, 2013) has a dynamic formulation. Let `Q_t = ∫₀ᵗ ẋ_s ds` be the cumulative traded notional, and `I_t ∈ ℝ` be the OW market impact. The OW market impact is modeled as
`I_t = κ/τ ∫₀ᵗ e^(-(t-s)/τ) ẋ(s) ds`
`İ_t = κ/τ ẋ_t - 1/τ I_t`

**Figure 11.2:** Market impact over time. The dashed line is the permanent market impact of the unit flow. The sum of temporary and permanent market impacts.

With an initial condition `I(0) = 0`, and with constant rate of execution `ẋ_0 = Q/T`, this differential equation has a simple solution:
`I_t = κQ/τ (1 - e^(-t/τ))`
and the trading cost is given by
`C = ∫₀ᵀ dt I_t q`
`= (κT/v) [τ/T (1 - τ/T (1 - e^(-T/τ)))]`
`= κT [τ/T - (τ/T)² (1 - e^(-T/τ))] (Q_T/V)`
The formula of `c = Q_T/V` is the same form as Equation 11.6.

(Gatheral, 2010) The functions are
`f(ẋ) = σ sgn(ẋ) |ẋ|^(1/2)`
`G(t) = 1/√t`
The trading cost is
`C = κσ ∫₀ᵀ dt ẋ(t) ∫₀^(t) ds (ẋ(s)/√(t-s))`
Assuming constant rate trading `ẋ_0 = Q/T` shown over an interval `[0,T]`, i.e., `ẋ_0 = Q/T` ... yields our final result:
`C = (4/3) κσ √(Q³/T)`
`c = (4/3) κσ √(Q/T)`
The unit execution cost is `c = (4/3) κσ √(Q/T)`, and independent of `T`.

---

**11.2 Finite-Horizon Optimization**

We have a large class of models to choose from. Which one to use is a question that can be settled empirically. In the following, we use the general form of Equation (11.1) for market impact, and a term proportional to trade size for transaction costs.

Let us focus on a single security. The market-impact-related trading cost is, after integrating by parts,
`ΔP_k = z_k P_k = z_k P_0 + z_k P_1 + ... + z_k P_k`
`ΔP_k = z_k P_0 + z_k P_1 + ... + z_k P_k = ∫₀^(t_k) ż_s P_s ds`
The term `P_0ẋ_0` is not a decision variable, so we can ignore it in the optimization process. As a boundary condition on the optimization, we assume that we end flat, i.e., `x_N = 0`. The overall trading cost becomes `-∫₀^T μ_s dx_s`. Now, let us write the expected costs from trading for a single asset:
`Cost_{trade} = - ∫₀^T μ_s dx_s - ∫₀^T f(ẋ_s, G_s) dx_s`
`(transaction cost) (impact cost)`

The idea behind multi-period optimization is to use the current forecast for excess returns over time, and plan trades for the entire horizon. Only the front-end of the trade is executed. In the next trading period, we develop or receive a new forecast, and we optimize again, using the updated portfolio as initial condition. This approach provides both stability and feedback. The stability comes from the fact that we use a long-term forecast, and the feedback from the fact that the functions will allow us to account for the decay in impact. In addition, we can also include as many side constraints as we want. In this section, we only include linear constraints, since they are the most common ones (e.g., constraints on factor risk, turnover, maximum position size, etc.). We have to solve numerically an optimization, whose execution time may introduce delays in trading and therefore adversely affect profitability. The second one is that the convergence properties of the problem depend on the forecast. Specifically, we need to ensure that the objective function is concave. This depends on the choice of market impact, which can make the problem non-concave. The third one is that we are incorporating the change in expected returns as a function of the horizon, but are not incorporating in the problem the dynamics of updates in our return forecast from one period to the next.

We split the trading periods into intervals delimited by timesteps `T_i`. We use the convention
`Δ_i := T_i - T_{i-1}`
For example, `Δ_k` may be in minutes, `Δ_N` may be in hours. `Δ_N` could be one day, `Δ_1` could be one week and `Δ_0` could be four weeks. At time `T_k`, we hold `x_k` dollars of the security. In interval `[T_i, T_{i+1}]` we have expected return `μ_i` and trade at rate `ẋ_i`. The relationship between `x_i` and `ẋ_i` is
`x_i = x(T_i)`
`ẋ_i = (x_i - x_{i-1})/Δ_i`
At time `t ∈ [T_i, T_{i+1}]`, we hold
`x(t) = x_i + ẋ_i(t - T_i)`
dollars in the security. This equation assumes we are not adjusting our holdings with price changes. The correct formula would be `ẋ_i = ż_i P_i`. In the words of Boyd et al. (2018), we are ignoring “second-order terms” in the holdings. This may be reasonable on two grounds. First, for short-enough periods, `P_i ≈ P_0`. Second, because we are not including the impact of `ẋ_i` on `P_i`, which is smaller than other errors already present in the model. For example, we replaced the realized market impact with the expected impact. The initial condition `x_0` and the trading rates `ẋ_i` determine `x_N`. We can then express `PnL_{total}` as a function of `ẋ_i`, and add this term to the objective function. The rest is (hairy) details.
`(transaction cost) = - Σ_{i=1}^N μ_i Δx_i`
`(impact cost) = - Σ_{i=1}^N f(ẋ_i, G_i) Δx_i`
`a_{i,j} > 0` if `i ≠ j`, `a_{i,j} < 0` if `i = j`
The variance penalty is integrated over a one interval.
`Var(x_N) = ∫ x_Nᵀ Ω x_N ds`
`Var(x_N) = Σ_{i,j} Δx_i Δx_j Ω_{i,j} + Δx_iᵀ Ω_{ii} Δx_i + (Δx_i + Δx_{i+1}) ...`
Now we can write the optimization formulation. We use the following notation:
`X = [x_1ᵀ, ..., x_Nᵀ]ᵀ`
`f(x) = -μᵀx - f_I(x)`
We also introduce `n×1` pairs of matrices and vectors `H_k ∈ ℝ^{m_k×n}`, `h_k ∈ ℝ^{m_k}`. These objects store linear constraints on holdings and portfolios for each stage of the optimization problem.

`max_{x_0, ..., x_N} Σ_{k=0}^{N-1} μ_kᵀ(x_{k+1} + x_k)` (expected PnL)
`- Σ_{k=0}^{N-1} A_kᵀ |x_k|` (trading cost)
`- Σ_{k=0}^{N-1} Σ_{j=k|j≠k}^{N-1} x_kᵀ Ω_{k,j} x_j` (impact cost)
`- (1/2) Σ_{k=0}^{N-1} [x_kᵀ Ω_{k,k} x_k + ... + (λ/2)ᵀΩ_N x_N]` (variance penalty)
`s.t. x_k - x_{k-1} - Δ_k = 0` (flow conservation)
`H_k x_k = h_k` (side constraints)
`x_k ∈ ℝ^n, k = 0, ..., N-1`
We assume that the initial position `x_0` is given, and usually (but not necessarily) we set a final holding condition `x_N = 0`. The “not necessarily” qualification depends on the parameters used in the problem. For example, if we set `μ_N = 0`, the optimization will attempt to reduce the size of the portfolio in the last stage, since holding the position incurs a risk cost. For large variance penalties and flat expected returns, the optimizer will sell a lot of `x_N`, that is, obtain `x_N < 0`, without the need of boundary condition.

---

**11.3 Infinite-Horizon Optimization**

We now present an infinite-horizon optimization model, introduced by Litterman et al. (2003). Compared to the finite-horizon optimization model of the previous section, the model allows only for quadratic costs and cannot accommodate generic side constraints `H_k x_k = h_k`. On the other side, it is flexible with respect to alpha processes.

We maximize a mean-variance objective, inclusive of transaction costs.
`(11.7) max_{ẋ} ∫₀^∞ e^(-ρt) [ μ_tᵀẋ_t - (1/2)ẋ_tᵀCẋ_t - (1/2)x_tᵀΩx_t ] dt`
where:
1.  The asset expected returns are described by a non-anticipative, stochastic process `μ_t` taking values in `ℝ^n`, defined over a probability space `(Ω, F, P)`.
2.  The matrix `C ∈ ℝ^{n×n}` is positive-definite and diagonal.[5] The cost rate from trading `ẋ_t` at time `t` is `(1/2)ẋ_tᵀCẋ_t`.
3.  The return covariance matrix `Ω ∈ ℝ^{n×n}` is positive-definite, with positive penalty factor `ρ`.
The optimal trading policy is described in Procedure 11.1.

**Procedure 11.1: Infinite-horizon optimal trading policy**
**Input:**
symmetric positive-definite cost matrix `C ∈ ℝ^{n×n}`;
symmetric positive-definite return covariance matrix `Ω ∈ ℝ^{n×n}`;
expected return process `μ_t` taking values in `ℝ^n`;
initial portfolio `x_0 ∈ ℝ^n`.
**Define:**
`(11.8) Γ := (C⁻¹Ω)¹/²`
`(11.9) b_t := ∫_t^∞ e^(-ρ(s-t)) Γ⁻¹ C⁻¹ E_s[μ_s] ds`
**Output:**
Optimal trading policy.
`x_t = e^(-Γt) (x_0 + ∫₀ᵗ e^(Γs) b_s ds)`
`ẋ_t = -Γx_t + b_t`
Optimum: `μ_t - Cẋ_t - (1/2)∫_t^∞ e^(-ρ(s-t)) Ωx_s ds = 0`

Let us try to interpret the (rather magical) objects we have introduced.
The `b_t` term of the vector `x_t` is a discounted expected return of asset `j`, where the discount factor is determined by the matrix `Γ`.
Consider the special case of uncorrelated asset returns, so that `C` and `Ω` are diagonal. The greater the volatility or the risk aversion, the larger the elements of `Γ`, thus the larger the discount factor.
On the other side, the higher the trading costs, described by `C_j`, the less we want to discount the future. Why? Because trading is expensive, we want to weight future returns more, so that we do not chase only instantaneous performance.
What is being discounted in `b_t` is not quite the future expected returns, but the alpha-normalized expected returns `C⁻¹E_s[μ_s]`. If unit costs are higher, we have a smaller future adjusted returns being discounted. However, we have a smaller `Γ`, hence we discount the future less. There are two competing effects. In Exercise 11.1, you will explore their relative impact. The term `Γ⁻¹` in `b_t` is the discounted version of the optimal trading policy.
`(11.10) b_t = (∫_t^∞ e^(-ρ(s-t)) (Γ⁻¹C)⁻¹ E_s[μ_s] ds)`
The optimal trading policy is recursive. The next optimal portfolio is a linear combination of the existing portfolio, discounted using matrix `Γ` and of the “alpha-to-go” `b_t`.

**Exercise 11.1 (The Impact of Costs on Trading).**
Consider the following simple problem. There is only one asset with volatility `σ` and cost parameter `c`. Consider two cases:
1.  The signal’s strength `μ` is exponentially decaying over time: `μ_t = μ_0 e^(-λt)`, with `λ > 0`.
The optimal strength is constant in interval `[0,T]`.
Prove that `b_0(c)` is a decreasing function of `c` in both cases. Generalize this result.

**11.3.1 Comparison to Single-Period Optimization**
Let us use the single-period problem. Define
`(11.11) Max E_k[μ_{k+1}ᵀ(x_{k+1} - x_k) - (1/2)(x_{k+1} - x_k)ᵀC(x_{k+1} - x_k) - (1/2)x_{k+1}ᵀΩx_{k+1}]`
`(11.12) Max E_k[μ_{k+1}ᵀx_{k+1} + C x_kᵀx_{k+1} - (1/2)x_{k+1}ᵀ(C+Ω)x_{k+1}]`
`(11.13) x_{k+1} = (C+Ω)⁻¹(E_k[μ_{k+1}] + C x_k)`
In the case `ρC⁻¹Ω ≪ I`, we can approximate
`x_{k+1} ≈ (I + ρC⁻¹Ω) C⁻¹ E_k[μ_{k+1}]`
The solution is similar to multi-period, in that it is a combination of the existing portfolio and an alpha-related term. If we assume that `Ω = 0`, i.e., no risk aversion, and `ρC⁻¹Ω ≪ I`, we approximate the first term with `x_k` and the multi-period solution is
`x_{k+1} ≈ (I - (ρC⁻¹Ω)²) C⁻¹ E_k[μ_{k+1}]`
The two are identical, except that a square-root term appears in the multi-period approximation.

**11.3.2 The No-Market-Impact Limit**
Consider the case of vanishing market impact. We set `C = cI_n` and let `c ↓ 0`. When `Γ ≥ I` from Equation (11.8), we have
`b_t = ∫_t^∞ e^(-ρ(s-t)) E_s[μ_s] ds`
`≈ e^(-ρt) C⁻¹ μ_t`
`= Γ⁻¹ e^(-ρt) C⁻¹ μ_t |_{s=0}^{s=∞}`
`= Γ⁻¹ (e^(-ρt) C⁻¹ - I) μ_t`
`≈ Γ⁻¹ C⁻¹ μ_t`
and
`ẋ_N+1 = -Γx_N + b_N`
`x_N+1 = (I - ρC⁻¹Ω)⁻¹ C⁻¹ E_N[μ_{N+1}]`
`≈ (I - ρC⁻¹Ω)⁻¹ C⁻¹ μ_{N+1}`
`x_{k+1} = (I - ρC⁻¹Ω)⁻¹ C⁻¹ E_k[μ_{k+1}]`
In the limit `ρC⁻¹Ω → 0`, a solution exists if
`x_t = (ρC⁻¹Ω)⁻¹ C⁻¹ μ_t`
`= ρ⁻¹Ω⁻¹μ_t`
This is the solution to the single-period MVO problem in the absence of transaction costs. The optimal solution is to rebalance instantaneously to the MVO allocation. Depending on the instantaneous alpha prediction. We have recovered the result from traditional single-period optimization.

**11.3.3 Optimal Liquidation**
Suppose that we hold a portfolio `x_0`, and have no forward-looking alpha: `μ_s = 0` for `s ≥ 0`. What is the optimal trading policy? In this case, `b_s = 0` and the optimal trade-value solution to the solution to the equation `ẋ_s = -Γx_s`. We reduce the position at an exponential rate, with the rate of liquidation depending on the matrix `Γ`. The larger the coefficient of risk aversion `ρ` and the volatility, the faster the liquidation. The higher the cost, the slower the liquidation.

**11.3.4 Deterministic Alpha**
Say that the future excess returns are a deterministic function `μ_s`. The function `b_t` is also deterministic and given by the integral (11.9). The solution to the Ordinary Differential Equation (ODE) (11.10) is
`b_t = ∫_t^∞ e^(-Γ(s-t)) C⁻¹μ_s ds`
`x_t = e^(-Γt) (x_0 + ∫₀ᵗ e^(Γs) C⁻¹μ_s ds)`
It is useful to present an indicative case of a “spiked” alpha: `μ_s(t-s_0) = μ_0 f(t-s_0)`. In this case the function `b_t` takes a simple form:
`b_t = 1/(t_0-t) ∫_t^(t_0) e^(-Γ(s-t)) C⁻¹μ_0 ds`
For `t ≤ t_0`,
`x_t = e^(-Γt) (x_0 + ∫₀ᵗ e^(Γs) Γ⁻¹ C⁻¹μ_0 ds)`
`= e^(-Γt) (x_0 + Γ⁻¹ cosh(Γt) Γ⁻¹ C⁻¹μ_0)`
In the formula above, we have introduced a direct extension of the hyperbolic cosine to square matrices, i.e.,
`sinh(X) := (e^X - e^(-X))/2`
When `t → ∞`, the optimal portfolio position is
`x_∞ = e^(-Γt_0) (x_0 + Γ⁻¹ cosh(Γt_0) Γ⁻¹ C⁻¹μ_0)`
For `t > t_0`, the portfolio is liquidated in the absence of alpha.

**11.3.5 AR(1) Signal**
Let us consider first the case of autoregressive expected returns:
`(11.17) μ_{t+1} = Φμ_t + η_t`
where
`Φ` is a diagonal matrix with `Φ_ii ∈ C, |Φ_ii| < 1`.
`η_t ~ N(0, Σ_η)` with `Σ_η` diagonal and positive-definite.
`η_t` are jointly independently time-dependent, i.e., `E_s[η_t] = η_s` for all `s`.
`E_t[η_s] = η_s` for all `s ≤ t`.
By repeated substitution we have
`(11.18) μ_{t+k} = Φ^k μ_t + Σ_{i=0}^{k-1} Φ^i η_{t+k-1-i}`
`(11.19) E_t[μ_{t+k}] = Φ^k μ_t`
The long term inverse covariance of `μ_t` are `B_k = (I - Φ^k)` and `Ω_k = (I - Φ^k)⁻¹ Σ_η`.
The continuous-time solution is
`(11.20) Γ = [C⁻¹(ρI - ΦᵀC⁻¹Φ)]¹/²` (from Equation 11.20)
`b_t = ∫_t^∞ e^(-Γ(s-t)) C⁻¹Φ^(s-t)μ_t ds` (from ρ = 0)
`= (Γ + C⁻¹Φ)⁻¹ C⁻¹μ_t`
Define `Γ = USVᵀ` and `Φ = VDVᵀ`. We rewrite `b_t` as
`b_t = U (∫₀^∞ diag(e^(-s_i t), ..., e^(-s_n t)) e^(-AΦ^k) dt) Φ^k μ_t`
`∫₀^∞ diag(e^(-s_i t), ..., e^(-s_n t)) AΦ^k dt = A_0 ∫₀^∞ (e^(-s_i t) Φ_i^k) dt`
`= A_0 [ -1 / (s_i - log(Φ_i)) ]`
`< ∞`
Define a matrix `H` by
`[H]_{i,j} = (s_i - log(Φ_j))⁻¹`
Then
`b_t = U ( (VᵀC⁻¹V) ○ H ) μ_t`
The optimal trading policy at time `t`, given portfolio `x_t` and predicted returns `μ_t`, is
`(11.21) δẋ_t = (-Γx_t + Kμ_t) δt`
(liquidation) (investment)
`(11.22) K := U ( (VᵀC⁻¹V) ○ H )`
Equation (11.21) has an intuitive interpretation. In the absence of expected returns, liquidate the book using a trading rate proportional to position size. With non-zero expected returns, we combine the liquidation with a trade that is a linear function of the expected returns.

---

**11.4 Appendix**

**11.4.1 Proof of the Infinite Horizon Quadratic Problem**

We need a few definitions:
`(x,y) := ∫₀^∞ e^(-ρt) x_tᵀ y_t dt`
`||x||² := (x,x)`
`(Kx)(t) := ∫₀ᵗ x_s ds`
The adjoint operator of `K` is such that
`(K⁺x, y) = (x, Ky)`
In formulas:
`(Kx,y) = ∫₀^∞ e^(-ρt) [∫₀ᵗ x_s ds]ᵀ y_t dt`
`= ∫₀^∞ e^(-ρt) x_sᵀ [∫_s^∞ e^(-ρ(t-s)) y_t dt] ds`
`= ∫₀^∞ e^(-ρs) x_sᵀ [∫_s^∞ e^(-ρ(t-s)) y_t dt] ds`
`(K⁺x)_s = ∫_s^∞ e^(-ρ(t-s)) x_t dt`
We solve Equation (11.7):
`max_{ẋ} ∫₀^∞ e^(-ρt) [ μ_tᵀẋ_t - (1/2)ẋ_tᵀCẋ_t - (1/2)x_tᵀΩx_t ] dt`
Introduce the variable `u` such that `x = Ku`
`max_{u} E_0[μᵀu] - (1/2)uᵀCu - (1/2)(Ku)ᵀΩ(Ku)`
FOC on `u`: `μ - Cu - K⁺ΩKu = 0`
`K⁺μ = CK⁺u + K⁺ΩKu`
`E_s[x_t] = E_s[x_s]`. The tower property of expectation is `E_s[E_t[X]] = E_s[X]` for `s < t`. Let `X_s,t = E_s[X_t]`. From the definition `X_s,t = X_s`.
Apply `E_s` to Equation (11.22) (assuming (11.22) is `μ_t - Cẋ_t - Ωx_t = 0` in continuous time):
`E_s E_t μ_t ds = C d/dt E_s x_t + ρE_s x_t + ∫_s^∞ E_s Ωx_u du` (This line seems to be a derivation step)
`d/dt E_s μ_t = C d²/dt² E_s x_t + ρ d/dt E_s x_t + ΩE_s x_t`
This is a linear ODE in `x_s,t`, which we can solve analytically. First, define
`Γ := (C⁻¹Ω)¹/²`
`b_t := ∫_t^∞ e^(-Γ(s-t)) C⁻¹E_s[μ_s] ds`
The solution that satisfies `lim_{t→∞} x_s,t = 0` is
`(11.23) x_s,t = e^(-Γ(t-s))x_s + ∫_s^t e^(-Γ(t-u)) b_u du`
from which it follows directly
`d/dt x_s,t |_{t=s} = -Γx_s + b_s`
Finally, from `dx_s/dt = ẋ_s` and `ẋ_s = u_s`, the law for the optimal trading policy follows:
`(Left) ẋ_s = -Γx_s + b_s`
`(11.24) x_t = e^(-Γt) (x_0 + ∫₀ᵗ e^(Γs) b_s ds)`
The value of the objective function is:
`∫₀^∞ (μ_tᵀẋ_t - (1/2)ẋ_tᵀCẋ_t - (1/2)x_tᵀΩx_t) dt`
`= ∫₀^∞ (μ_tᵀ(-Γx_t+b_t) - (1/2)(-Γx_t+b_t)ᵀC(-Γx_t+b_t) - (1/2)x_tᵀΩx_t) dt`
`(11.25) = (μ_0ᵀx_0 + b_0ᵀCT⁻¹x_0 - (1/2)x_0ᵀΩx_0 - (1/2)b_0ᵀCb_0)`

---

**The Takeaways**

1.  Trading incurs various costs, including market impact, which can turn profitable strategies unprofitable.
2.  Market impact is the cumulative market response to new order flow, influenced by inventory reduction, informational effects, mimetic effects, and strategic trading.
3.  Total transaction costs comprise spread costs, temporary market impact, and permanent market impact.
4.  Spread costs arise from the bid-ask spread; market orders cross the spread and remove liquidity.
5.  Temporary impact is the short-term price change during post-execution; temporary impact occurs during and immediately after execution.
6.  Temporary market impact is modeled using functions `f` (instantaneous impact) and `G` (propagator), representing price impact dynamics.
7.  Models like Almgren-Chriss, Kyle, Obizhaeva-Wang, and Gatheral use specific forms of `f` and `G` to describe impact.
8.  Finite-horizon optimization plans trades over multiple periods, accounting for market impact, transaction costs, and risk constraints.
9.  Multi-period optimization is flexible but may require complex numerical solutions and has convergence concerns.
10. Infinite-horizon optimization provides analytical optimal trading policies by maximizing a mean-variance objective with transaction costs.
11. The optimal policy balances expected returns, trading costs, depending on initial risk aversion and costs.
12. Special cases like no-market-impact limit and optimal liquidation illustrate the policy under different conditions.

---

**Notes**

1.  The academic literature on optimal execution is vast, and growing rapidly. This chapter offers a selective treatment of portfolio optimization. Reference texts are Gârleanu (2009); Bouchaud et al. (2018); Bacidore (2020); Velo et al. (2020); Moallemi (2021).
2.  There are at least two classes of costs associated with it. First, there are opportunity costs. If the price moves away, we may be forced to reprice the order and pay a higher price than the market order we could have submitted. Second, and counterparty is typically an informed trader who is forcing a market order to reveal their expectation on our side.
3.  A market order is an order to buy or sell a stock at the best available price offered by a counterparty, and that therefore removes inventory.
4.  `δ(.)` is a “pulse” generalized function, zero everywhere except the origin, and whose integral is 1.
5.  The argument on dimensional analysis, see Bussan and Ramaj (2019); Toth et al. (2011); Gatheral (2010); Cont (2011); Mahajan (2012).
6.  Linear equality constraints can be priced in the objective function, but linear inequality constraints can’t.
7.  A looser description of such a process is that the value of `μ_k` is known based on the information available at time `t`. The rigorous definition of non-anticipative (or adapted) process is beyond the scope of this book; see Shreve (2004a) for a more satisfactory such treatment (Vol I).
8.  We make the assumption of diagonal cost matrix because it is intuitive and most relevant to applications. It can be relaxed to model cross-market impact among securities. Many such models use quadratic costs. See, e.g., Mastromatteo et al. (2017); Alfonsi et al. (2022).
   

Okay, here is the Markdown compilation for Chapter 12.

```markdown
Everand

