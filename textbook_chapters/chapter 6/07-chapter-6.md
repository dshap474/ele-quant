**Chapter 6**
**Fundamental Factor Models**

**The Questions**

1.  What are the steps involved in the estimation of a fundamental factor model?
2.  How do we handle heteroskedastic returns?
3.  Can we estimate a model when we have “redundant” characteristics of the securities in the loadings matrix?
4.  What should we be careful about when estimating the factor covariance matrix?
5.  Can we make the model more responsive to sudden changes in volatility?
6.  How do we estimate off-diagonal terms in the idiosyncratic covariance matrix?
7.  How do we link different factor models into a coherent one?
8.  How do we model currencies in the model, and how do we change the reference currency?

Fundamental (or characteristic-based) factor models estimate Equation (4.1) using as inputs $\mathbf{R}_t$ and $\mathbf{B}_t$. The outputs of the models are estimates of factor and idiosyncratic returns $\hat{\mathbf{f}}_t, \hat{\mathbf{\epsilon}}_t$, as well as their covariance matrices $\hat{\Omega}_f, \hat{\Omega}_\epsilon$. Fundamental factor models are perhaps the most popular ones among practitioners. Reasons for their popularity are:

*   Good Performance. Commercial models are the outcome of a long process of refinement. The first models date back to the mid-1970s. Consequently, some important factors have been identified.
*   Interpretability. Firm characteristics provide a summary description of individual assets, and exposures based on these characteristics give a measure of systematic risk.
*   Connections to Academic Research. In the asset pricing literature, multifactor models originate with the Arbitrage Pricing Theory of Ross (1976). In fact, the reference model used by academic researchers to explain pricing anomalies is the three-factor model by Fama and French (1993).
*   Alpha Research. Fundamental models are the workhorse of alpha research, because they allow the portfolio manager to incorporate almost any data source, to analyze very large datasets, to interpret the results of their analysis, and to link the outcome to a portfolio construction system.

---

**6.1 The Inputs and the Process**

There are two major steps needed to identify a factor model. Some of them require involved quantitative methods; others are more art than science. Before we even begin to describe the steps, we should focus on the inputs.

**6.1.1 The Inputs**

Fundamental model inputs are:
1.  A set of returns per asset/date, i.e., the $\mathbf{R}_t$ part.
2.  A set of raw characteristics per asset/date/characteristic identifier. Asset characteristics are generally denoted by $\mathbf{B}_t$.

Asset returns are usually computed over intervals of equal duration. These intervals determine the periodicity of the model. Daily returns may be based on close-to-close prices. Intraday returns may be based on the last transaction price at intervals that divide the trading day. The interval can range from 30 minutes to sub-second intervals. It should make sense to use returns over the shortest interval, but this is not the case. The answer to the question “what is the final price at a time period?” is not easy or unique. Ultimately, models of returns should help the portfolio manager develop a profitable real-life strategy. If prices are such that we cannot trade on them, or if transaction costs at the trading horizon are too high, then the model will not be useful, no matter how well it fits the data on paper. Consider the closing price. Where does it come from? In many stock exchanges, at the end of the day the limit-order book (LOB) is replaced by a closing auction (CA). Without delving into the details of a CA, it suffices to know that a CA is a very different beast from a LOB. If the model is based on daily returns, and the final price is the closing auction price, it is meaningful, in the sense that it is exploitable by a portfolio manager, at a non-negligible size. Now, compare this scenario to one in which we are interested in modeling a small-cap stock that trades very little. In the extreme case, such a stock would not trade at all for long stretches of time. If it is a member of the estimation universe, then the model will use its zero returns at any volume. In addition, if we model intraday returns, then we must exercise additional care. What does it mean that the price at the end of a 5-minute interval was $6.35 per share? Maybe the stock did not even transact in that interval, and the price was stale. Or maybe it transacted once, but only a handful of shares changed hands. The transaction price was $6.35$. Maybe there were many transactions, but the transaction happened at the ask, but the transaction just before happened at the bid. Or should we use the mid-price between the bid and the ask? Could our strategy actually trade at either of these prices in real life? These are just some of the many questions that arise when dealing with high-frequency return data. Daily returns, but not only for many asset classes, determining good daily closing prices is a very challenging, important and thankless task. The details are

(Right Sidebar from Page 212)
two asset- and data-vendor-specific to be covered in detail; moreover, this is an area where biases accumulate. The first impurity is that any material to their business. A fix is needed. The second one is that the data are noisy. The third one is that the problem is how to solve this already in Section 2.1.4. The fourth one is that the model what price is appropriate (bid, ask, mid); whether to explicitly model the noise in price observations; whether time-of-day non-stationarity matters; whether to model transaction costs explicitly; whether to model market impact. The fifth one is that returns are usually non-stationary. The sixth one is that to avoid looking-ahead bias, we need to use returns that are available at time $t-1$ and model in a period $t$. Liquidity. For our purposes, we could be pressed by trading volume in a period. Usually, average daily trading volume is available. Liquidity is related to price discovery. The price of a very liquid stock is less prone to manipulation, and it is more likely to reflect the true value of the asset. There are choices a modeler makes. Several choices of model inputs are related. Unless you want to model market microstructure explicitly, and want to rely on closing period prices, then a shorter period will imply that your model universe will be smaller.

Raw characteristics data. This is the “art”, or better, the “dark art” part of the modeling task. “Raw data” can mean almost anything. A possible classification of raw data is structured versus unstructured. The former includes numerical data and categorical data that can be converted to a numerical factor or an estimation period. Categorical data take only a finite set of values, which do not necessarily have an ordering relationship. Examples are the country and sector of a stock. A slightly less common example is the credit rating of a company, which does admit an ordering. The latter includes textual data, such as news or social media, and tabular form. Examples are the earnings transcripts of a company, or its regulatory filings (Forms 10-K or 10-Q), or its web search, or a web scraping of a firm with information about its products, or the consumer credit card transactions of a firm; a list could go on and on. Some of these data are more useful than others for modeling purposes, some are just a ruse of the vendor, or of the modeler. For asset return modeling purposes, we extract from these vast troves of alternative data some representative statistics that can be interpreted as structured data. For example, from transaction data, we can extract flows (dollars transacted in a stock); from textual data, we can extract sentiment (positive, negative, or neutral change at the company level); or we can extract measures of geographical dispersion. Some of these operations can be automated; some may require the use of machine learning tasks like classification and clustering. One important feature of all these operations, though, is that they should result in data that are comparable across assets and that admit a natural interpretation of security returns as returns in excess of a rate received in the absence of investing. In the United States, this requires the highest amount of human intelligence. A great number of papers have been published on characteristic data. Maybe in the future we will be able to feed such disparate sources of information into a black box and directly produce portfolio return recommendations. In that event, I’ll gladly write a second edition to this book.

---

**6.1.2 The Process**

The estimation steps of a characteristic model are:
1.  Data ingestion. This step encompasses receiving datasets from vendors, checking their integrity, and performing essential data checks. These include:
    *   Ensure that data are of the correct type and not corrupt. This happens with positive probability.
    *   Ensure that the set of securities is not substantially different from the previous period.
    *   Ensure that the fraction of missing data per asset and characteristic is not substantially different.
    *   Identify and report data outliers.
2.  Estimation universe selection. I introduced issues related to this set earlier in this chapter. The main criteria are:
    *   Tradability. The assets must be sufficiently liquid, because FMPs include them all.
    *   Data quality. This is closely related to the liquidity of the assets, but for a different reason. We need securities for which price discovery is good, i.e., assets that have a high informational value, since we are using their prices for return calculations and model estimation.
    *   Relevance to investments. The estimation universe should be overlapping to some extent with the investment universe of the strategy. This is a judgment call. There is not to my knowledge a rigorous treatment of this problem.
3.  Winsorization. Identify outliers in returns of the estimation universe and winsorize them.
4.  Loadings generation. Generate characteristics $\mathbf{B}_t$ by transforming and combining raw data.
5.  Cross-sectional regression. For each $t=0, \dots, T$, perform a cross-sectional regression of asset returns $\mathbf{R}_t$ against the loadings $\mathbf{B}_t$. The outputs of this step are the estimates $\hat{\mathbf{f}}_t$ and $\hat{\mathbf{\epsilon}}_t$. Furthermore, from these, using the time series from the previous steps estimate:
    *   the factor covariance matrix $\hat{\Omega}_{f,t}$
    *   the idiosyncratic covariance matrix $\hat{\Omega}_{\epsilon,t}$
    *   the risk-adjusted performance of factor returns.

The next three sections address specific aspects of model inputs, regression, and covariance estimation. We then consider estimation in Section 6.5. The next section presents an advanced topic on linking multiple factor models coherently, and also considers portfolio assets denominated in multiple currencies. Finally, we offer a preliminary description of identifying the factor characteristics.

---

**6.2 Cross-Sectional Regression**

The first step is cross-sectional regression.
(6.1)
$$ \mathbf{r}_t = \mathbf{B}_t \mathbf{f}_t + \mathbf{\epsilon}_t, \quad t \in N $$
where the parameters to be estimated are $\mathbf{f}_t$ and $\mathbf{\epsilon}_t$. This is a case of random design: the tuple $(\mathbf{R}_t, \mathbf{B}_t, \mathbf{f}_t, \mathbf{\epsilon}_t)$ can be viewed as independent samples drawn from a common distribution. We observe $(\mathbf{R}_t, \mathbf{B}_t)$.
Several regression approaches are possible. One may minimize the square loss function $\| \mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t \|^2$. The assumptions behind this step are:
1.  The matrix $\mathbf{B}_t \in \mathbb{R}^{N \times K}$ has full rank. A necessary but not sufficient condition for this is that $m \le N$.
2.  Residual returns $\mathbf{\epsilon}_t$ have zero mean, are homoskedastic (i.e., $\text{Var}(\epsilon_{it}) = \sigma_\epsilon^2$), and are uncorrelated, and independent of each other.
3.  Factor returns residual returns are independent of each other.
4.  Factor and residual returns are “well-behaved”, in the sense of having at least finite fourth moments.
These assumptions can be relaxed in various ways. The matrix $\mathbf{B}_t$ is often rank-deficient. Later in this section we will introduce ways to deal with rank-deficient matrices. In order to have a unique solution. Homoskedasticity is also not a necessary assumption. If residuals have different variances for different assets, we can weight the observations by the inverse of their variance. Heteroskedasticity is a fact of life. In all of these cases, we use weighted least squares, so that the single term does not dominate the sum of losses, and unduly affect the parameters’ estimation.
We estimate Model (6.1) by minimizing the sum of a loss function $L: \mathbb{R}^N \times \mathbb{R}^N \to \mathbb{R}^+$:
$$ \min_{\mathbf{f}_t} L(\mathbf{R}_t, \mathbf{B}_t \mathbf{f}_t) $$
In this section, we choose to minimize the weighted sum of the squared residuals. We know a diagonal, positive matrix $\mathbf{W}$ whose diagonal terms can be interpreted as weights assigned to observation (i.e., asset) $j$. We then find $\mathbf{f}_t$ that minimizes
(6.2)
$$ L(\mathbf{R}_t, \mathbf{B}_t \mathbf{f}_t) = (\mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t)^T \mathbf{W} (\mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t) $$
There are good reasons for this choice. If we assume that Model (6.1) is the true model, then least squares gives us the lowest-variance unbiased estimate among all the linear models (Pearson, 2022). The lack of bias matters for performance attribution and alpha identification. Even a small bias in factor return estimation (and, consequently, in residual returns) would accumulate over the course of a year. The low variance of the estimate matters for risk estimation. The additional benefit of weighted least squares regression is that its estimates have a natural interpretation in terms of Factor-Mimicking Portfolios (FMPs). We will cover these in detail later. For now, it should suffice to say that FMPs are investable portfolios whose returns track as well as possible the true (but unobserved) factor returns. Our estimation procedure is to use this set as a starting point, and to run through diagnostics to identify its possible shortcomings. To do so, we make the assumption that the model is $\mathbf{R}_t = \mathbf{B}_t \mathbf{f}_t + \mathbf{\epsilon}_t$, with $\mathbf{\epsilon}_t \sim N(0, \Omega_\epsilon)$. We are addressing the problem of estimating $\mathbf{f}_t$ by minimizing both sides of (6.2). Now assume that the idiosyncratic covariance matrix is diagonal, $\Omega_\epsilon = \text{diag}(\sigma_{\epsilon,1}^2, \dots, \sigma_{\epsilon,N}^2)$. We use the Ordinary Least Squares (OLS) loss function (i.e., $\mathbf{W} = \mathbf{I}_N$ in (6.2)), which is equivalent to the loss function in Equation (6.1) with a weight matrix $\mathbf{W} = \Omega_\epsilon^{-1}$.
Factor loadings are assumed to be constant over time. This simplifies the formulas below, but can be relaxed by simply regressing the returns on the time-varying loadings.
Given $\mathbf{B}_t, \Omega_\epsilon$, the Gaussian likelihood is given by
$$ L = \prod_{t=1}^T (2\pi)^{-N/2} |\Omega_\epsilon|^{-1/2} \exp \left( -\frac{1}{2} (\mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t)^T \Omega_\epsilon^{-1} (\mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t) \right) $$
If we denote the matrix of returns $\mathbf{R} \in \mathbb{R}^{N \times T}$ and $\mathbf{F} \in \mathbb{R}^{K \times T}$, the log-likelihood is equivalent to $-\frac{NT}{2} \log(2\pi) - \frac{T}{2} \log |\Omega_\epsilon| - \frac{1}{2} \sum_{t=1}^T (\mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t)^T \Omega_\epsilon^{-1} (\mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t)$. We write the optimization problem as
$$ \min_{\mathbf{F}} \frac{1}{2} \| \Omega_\epsilon^{-1/2} (\mathbf{R} - \mathbf{B}\mathbf{F}) \|_F^2 $$
$$ \text{s.t. } \mathbf{F} \in \mathbb{R}^{K \times T} $$
We consider first the case of a single period. In this case $\mathbf{R}$ and $\mathbf{F}$ are column vectors. The solution is formally given by
$$ \hat{\mathbf{f}}_t = (\mathbf{B}_t^T \Omega_\epsilon^{-1} \mathbf{B}_t)^{-1} \mathbf{B}_t^T \Omega_\epsilon^{-1} \mathbf{R}_t $$
In the case of multiple periods, the problem is the sum of the single-period problems.
$$ \mathbf{R} - \mathbf{B}\mathbf{F} = \sum_{t=1}^T (\mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t)^T \Omega_\epsilon^{-1} (\mathbf{R}_t - \mathbf{B}_t \mathbf{f}_t) $$
Each term can be minimized independently. Hence, we have
(6.3)
$$ \hat{\mathbf{f}}_t = (\mathbf{B}_t^T \Omega_\epsilon^{-1} \mathbf{B}_t)^{-1} \mathbf{B}_t^T \Omega_\epsilon^{-1} \mathbf{R}_t $$
As a direct extension of the previous formula in matrix form, the problem of minimizing $\| \mathbf{A} - \mathbf{B}\mathbf{X} \|_F^2$ has a closed-form solution:
(6.4)
$$ \arg \min_{\mathbf{X}} \| \mathbf{A} - \mathbf{B}\mathbf{X} \|_F^2 = (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T \mathbf{A} $$

**Insight 6.1: Which idiosyncratic covariance matrix?**
In the cross-sectional regression procedure, we have taken as an input the covariance matrix $\Omega_\epsilon$. However, this matrix is an output of the factor estimation procedure. How to resolve this circularity? One possibility is to use the identity matrix as a proxy for the true idiosyncratic covariance matrix. This proxy won’t be the best possible covariance matrix, but it is better than the identity matrix, which corresponds to OLS. The first, and simplest one, is to take advantage of a previous estimation of the covariance matrix, say from BARRA or Axioma. A second approach is to perform a two-stage process. In the first stage, we use the identity matrix as a proxy in the cross-sectional regression, and estimate the idiosyncratic covariance matrix, as explained in Section 6.4. In the second stage, we use this estimate in the weighted estimation process, starting with the cross-sectional regression, and ending with a final idiosyncratic covariance matrix. Finally, there is a variant on this second approach. We can use the identity matrix in the first stage, and average returns only for a number of periods. In the following example, we have 20 years of history, we could perform the two-stage process in the first year only. For every day after the first year, we can use as proxy the final estimate of the idiosyncratic covariance matrix estimated for the previous day.

**6.2.1 Rank-Deficient Loadings Matrices**

In some cases the loadings matrix is rank-deficient: even if there are $K$ factors, the number of independent columns is $m < K$. As concrete (and very common) examples, consider the following:
1.  There is a factor with loadings for each asset equal to 1. This is sometimes called a “country,” “region,” or “universe” factor, since all assets are identically affected by changes in this factor. The interpretation is that this factor is a “market” factor. In this case, the regression matrix $\mathbf{B}_t$ has at least one linear dependency: the sum of the last $m$ columns is equal to the first column, which can be interpreted as non-negative weights summing to one, for simplicity, assume that the first factor is the country, and the next $m-1$ are industries. Then the vector
    $$ \mathbf{v} = (1, -1, \dots, -1)^T $$
    is such that $\mathbf{Bv} = \mathbf{0}$. The matrix is rank-deficient.
2.  In most multi-country models there are industry as well as country factors. Say that $N_{ind}$ and $N_{cty}$ are industry loadings, followed by $N_{cty}$ country loadings. Then the vector
    $$ \mathbf{v} = (1, 1, \dots, 1, -1, -1, \dots, -1)^T $$
    where the first $N_{ind}$ ones are ones and the remaining $N_{cty}$ are negative ones, also satisfies $\mathbf{Bv} = \mathbf{0}$.
We generalize this to the case where there are $m-p$ independent vectors $\mathbf{v}_k$ such that $\mathbf{Bv}_k = \mathbf{0}$. Because of this, $(\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B})$ is not invertible and it is not possible to estimate $\mathbf{f}$ using Equation (6.3). There are at least three ways to address such an issue:
1.  The first is to remove the redundancy. For example, remove the country factor and/or $p$ industry or country factors from the regression. The drawback is that the original loadings matrix is easier to interpret. We would like to know a portfolio exposure to the country and to all industries. The second drawback is that, if the industries’ exposures are positive, which is often true, it implies that the industries’ exposures are small. And, of course, all industries are useful. Just ask the portfolio manager whose main covered industry was removed from the model.
2.  The second is to add a small quadratic penalty term to Equation (6.2). The term is $(\mathbf{f}^T \mathbf{D} \mathbf{f})$, where $\mathbf{D}$ is a diagonal matrix of positive constants. The factor estimates are no longer unbiased (or the linear model), so a careful analysis would be needed before using this method. In the limit $\delta \to 0$, the solution to the regression problem is the ridge regression
    $$ \hat{\mathbf{f}} = (\mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{B} + \delta \mathbf{I}_K)^{-1} \mathbf{B}^T \Omega_\epsilon^{-1} \mathbf{R} $$
    , where $^+$ denotes the Moore-Penrose pseudo-inverse operator. Finally, we can add $m-p$ side constraints of the form $\mathbf{C}^T \mathbf{f} = \mathbf{c}$. An example is that the sum of industry factor returns is zero. Our regression problem then finds some person complexity to the estimation process, but maintains or even enhances the interpretability of factor returns. For example, we may require that the market-weighted sum of industry factor returns be zero. This would mean that $\mathbf{w}_{mkt}^T \mathbf{B}_{ind} \mathbf{f}_{ind} = 0$, where $\mathbf{B}_{ind}$ is the submatrix of industry factor returns, and $\mathbf{w}_{mkt}$ is a weight vector of asset market caps per asset. The constraint says “the sum across assets of market-weighted industry returns must be zero.” If $\mathbf{w}$ are chosen to be the weight of a benchmark portfolio, this may read as “the benchmark portfolio must have no industry returns.”

---

**6.3 Estimating the Factor Covariance Matrix**

We have a random vector of factor returns $\hat{\mathbf{f}}_t$ from which we want to estimate $\Omega_f$. We assume that the $\hat{\mathbf{f}}_t$ has zero fourth moments, but unlike in Section 2.2.4, we cannot assume $\hat{\Omega}_f = \mathbf{I}_K$. By construction, we do not expect the matrix to be spiked. The number of samples over which we estimate the covariance matrix can be larger than the number of factors. For example, we could estimate a model with 10 factors and 300 days of estimation. The assumption that its estimate is $\hat{\Omega}_f = \frac{1}{T} \sum_{t=1}^T \hat{\mathbf{f}}_t \hat{\mathbf{f}}_t^T$.
Let
$$ \hat{\Omega}_f = \frac{1}{T} \sum_{t=1}^T \hat{\mathbf{f}}_t \hat{\mathbf{f}}_t^T $$
By the Law of Large Numbers, $\hat{\Omega}_f \to \Omega_f$ almost surely. Both eigenvalues and eigenvectors converge to the true population values. See the Appendix, Section 7.6.2. Factor volatilities converge to the true (aka denoted population) volatilities, and the principal components of the factor covariance matrix converge to their population counterparts, so long as the volatilities of factors are all sufficiently separated. This seems to settle the issue of covariance estimation: just take the empirical covariance matrix. There are several problems, though:
*   Oftentimes, the number of factors is not much smaller than the number of observations. In this case, shrinking may improve the quality of the estimate.
*   We will see that factor return estimates are influenced by the estimation process. This is another argument in favor of shrinkage.
*   Factor returns are non-stationary, sometimes dramatically so at the onset of a crisis. We need to take this into account.
*   Factor returns are serially autocorrelated. This is not correct.

**6.3.1 Factor Covariance Matrix Shrinkage**

The first issue is that the factor return estimates $\hat{\mathbf{f}}_t$ are just that: estimates. They are the outcome of WLS linear regression estimates, Equation (6.3). The covariance matrix of the estimates $\hat{\mathbf{f}}_t = (\mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{B}_t)^{-1} \mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{r}_t$. This implies that the covariance matrix of the estimates $\hat{\mathbf{f}}_t$ is biased.
(6.5)
$$ \text{var}(\hat{\mathbf{f}}_t) = \Omega_f + (\mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{B}_t)^{-1} $$
Insight 6.2: FMP, interpretation of factor covariance shrinkage
An alternative lens to interpret Equation (6.5) is via factor-mimicking portfolios. The return of FMP $\mathbf{w}_k$ is $\mathbf{w}_k^T \mathbf{r}_t = \mathbf{e}_k^T (\mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{B}_t)^{-1} \mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{r}_t = \hat{f}_{k,t}$. The estimated covariance of the returns of FMP is $\text{cov}(\hat{\mathbf{f}}_k, \hat{\mathbf{f}}_j) = (\mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{B}_t)_{k,j}^{-1}$. This suggests that we should shrink the empirical covariance matrix in order to obtain an unbiased estimate.
(6.6)
$$ \hat{\Omega}_f^u = \hat{\Omega}_f - (\mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{B}_t)^{-1} $$
How big is the correction? In the simpler but instructive case where $\mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{B}_t = \mathbf{I}_K$, i.e., the estimated factor returns $\hat{\mathbf{f}}_t = \mathbf{B}_t^T \mathbf{r}_t$,
$$ \text{var}(\hat{\mathbf{f}}_t) = T^{-1} \sum_{t=1}^T \hat{\mathbf{f}}_t \hat{\mathbf{f}}_t^T = \Omega_f + \mathbf{I}_K $$
, and therefore
(6.7)
$$ \hat{\Omega}_f^u = \hat{\Omega}_f - \mathbf{I}_K $$
In applications the number of observations ranges from 250 to 1000 days and the number of factors ranges from 10 to 50. The number of assets is in the order of 5000 and 30 years of daily returns. Therefore we are not always in the regime $N \ll T$ and the asymptotics of Section 7.6.2 do not apply, neither the results for spiked covariance matrices. A popular shrinkage applied to the covariance matrix is Ledoit-Wolf shrinkage (Ledoit and Wolf, 2003a,b, 2004). It has the advantage of being simple, robust, and having good performance. The shrunken covariance matrix is
$$ \Omega_{f,shrink}(\rho) = (1-\rho) \hat{\Omega}_f + \rho \frac{\text{trace}(\hat{\Omega}_f)}{m} \mathbf{I}_m $$
which we combine with Equation (6.7):
$$ \Omega_{f,shrink}(\rho) = (1-\rho)[\text{var}(\hat{\mathbf{f}}_t) - (\mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{B}_t)^{-1}] $$
$$ + \rho \frac{\text{trace}(\text{var}(\hat{\mathbf{f}}_t) - (\mathbf{B}_t^T \Omega_{\epsilon,t}^{-1} \mathbf{B}_t)^{-1})}{m} \mathbf{I}_m $$
where $\rho \in (0,1)$ is a tunable parameter.

---

**6.3.2 Dynamic: Conditional Correlation**

An alternative approach to estimating the empirical covariance matrix in Equation (6.5) is to model the factor volatilities and correlations separately. Normally, we decompose the population covariance matrix into the product of a correlation matrix $\mathbf{C}$ and a diagonal matrix $\mathbf{V}$ containing the factor volatilities:
$$ \Omega_f = \mathbf{V} \mathbf{C} \mathbf{V} $$
Bollerslev (1990) modeled the time-varying correlation and the correlation matrix as constant. Practitioners estimate the empirical correlation matrix and the volatility vector using exponentially weighted averages with different half-lives:
$$ \text{diag}(\mathbf{V}_t) = T_V^{-1} \sum_{s=0}^{T_V-1} K_V e^{-s/T_V} \hat{\mathbf{f}}_{t-s} \odot \hat{\mathbf{f}}_{t-s} $$
$$ \mathbf{C}_t = K_C^{-1} \sum_{s=0}^{T_C-1} K_C e^{-s/T_C} \hat{\mathbf{f}}_{t-s}^* (\hat{\mathbf{f}}_{t-s}^*)^T $$
where $T_V < T_C$ are half-lives for factor correlations and factor variances, respectively, and $K_V, K_C$ are normalizing constants. In many equity models estimated using daily returns, the half-lives are set between three months (for variances) to one or two years.

**6.3.3 Short-Term Volatility Updating**

Estimated factor returns often exhibit large, unanticipated volatilities. Anecdotally, their volatility does not vary smoothly but discontinuously, with regimes of high volatility followed by a quick transition to low-volatility regimes. This poses two challenges. First, the standard EWMA model for volatilities in Equation (6.8) will react too slowly to the sudden increases in volatility. The carryover effect of this is that the investors will severely underestimate systematic risk at the time when they need accurate estimates the most. Second, the estimates react too slowly to subsequent volatility declines. By the nature of the weights, previous volatilities decay too slowly than warranted, resulting in an overestimation of risk. Several approaches have been proposed to address this issue. We mention one that performs well and is simple: Short-Term Volatility Updating (STVU). First, we model the multivariate factor returns so that they are modulated by a latent state variable $z_t$:
(6.8)
$$ \hat{\mathbf{f}}_t = z_t^{1/2} \mathbf{C}_t^{1/2} \mathbf{\eta}_t $$
$$ \mathbf{\eta}_t \sim N(0, \mathbf{I}_m) $$
$$ z_{t+1} = \phi z_t + \sigma_z \epsilon_t $$
In the degenerate case where $\sigma_z=0$, i.e., $z_t$ is all 1, the model reduces to one where the factor covariance matrix is $\Omega_f = C_t$.
Define $e_t := \log |\mathbf{u}_t|^2 - \log \|\mathbf{\eta}_t\|^2$.
This is a linear state-space model. Define
$$ \kappa = E(\log \|\mathbf{\eta}_t\|^2) $$
$$ \epsilon_t := \log |\mathbf{u}_t|^2 - \kappa $$
$$ x_t := \log z_t $$
$$ y_t := \log |\mathbf{u}_t|^2 - \kappa $$
The state-space equation is
$$ x_{t+1} = \phi x_t + \sigma_z \epsilon_t $$
and the estimate of the state takes the form,[7]
$$ \hat{x}_{t+1|t} = (1-e^{-1/T_V}) \sum_{s=0}^{T_V-1} e^{-s/T_V} (\log |\mathbf{u}_{t-s}|^2 - \kappa) $$
From the first equation in Model (6.8), the factor covariance matrix is then adjusted by multiplying by $e^{\hat{x}_{t+1|t}}$. At the time when they need accurate estimates the most.
Some implementations use the linear approximation of this formula and the approximate equality
$$ \kappa = \log(m) + E(\log \|\mathbf{\eta}_t\|^2/m) \approx \log m $$
since $E(\log \|\mathbf{\eta}_t\|^2/m) \approx 0$ as $m \to \infty$.
(6.9)
$$ e^{\hat{x}_{t+1|t}/2} \approx \exp \left( (1-e^{-1/T_V}) \sum_{s=0}^{T_V-1} e^{-s/T_V} \frac{|\mathbf{u}_{t-s}|^2 - m}{m} \right) $$
The interpretation of the formula is clearer in the special case of uncorrelated factor returns. In this case, $m=1$. If we view the random variable $u_t$ as $N$ iid samples of a random variable, the term $|\mathbf{u}_t|^2/m$ gives us an estimate of its standard deviation, and if this estimate exceeds one, then our original estimate for the standard deviation of the $f_t$ needs to be scaled upward. That is what the model does. The half-life is typically chosen to be ten days for daily risk models, in order to incorporate the rapid onset of a shock.

**6.3.4 Correcting for Autocorrelation in Factor Returns**

Daily factor and asset returns usually exhibit mild, but non-zero, short-term autocorrelation. When the factor covariance matrix is estimated on shorter time intervals, the autocorrelation may be more pronounced. In these cases, correcting for autocorrelation may improve the model performance, at a cost. We build on previous work by Scholes and Williams (1977) and assume that the observed returns follow an autoregressive process of order $L_{max}$ that is a function of underlying uncorrelated returns. The coefficients in the AR($L_{max}$) equation are random, but sum to 1. Let the lagged covariance matrix $C_j$ be defined as
$$ [C_j]_{k,l} := \text{cov}(\hat{f}_{k,t}, \hat{f}_{l, t-j}) $$
Then the autocorrelation consistent estimator is given by
$$ \hat{\Omega}_f^c = C_0 + \sum_{j=1}^{L_{max}} (C_j + C_j^T) $$
An alternative approach, which is asymptotically consistent in the limit $T \to \infty$, is Newey and West’s estimator (Newey and West, 1987).

---

**6.4 Estimating the Idiosyncratic Covariance Matrix**

Next, we need to estimate the covariance matrix $\Omega_\epsilon$, based on the period estimated idiosyncratic returns $\hat{\mathbf{\epsilon}}_t$.

**6.4.1 Exponential Weighting**

As in the case of factor volatility, we use exponential weighting for idiosyncratic volatility estimation. Let $\hat{\sigma}_{\epsilon, it}^2 = E[\hat{\epsilon}_{it}^2]$ be the mean of estimated idiosyncratic returns for asset $i$ at time $t$. The weighting matrix is a diagonal positive-definite matrix $\mathbf{W}_t$. A common choice for the diagonal terms is $\mathbf{W}_{t,t} = \kappa \exp(-t/\tau)$.
; the positive constant $\kappa$ is such that the diagonal terms sum to $T$. The EWMA empirical idiosyncratic covariance matrix is then $\hat{\Omega}_\epsilon = \text{EWE}_t^T$.

**6.4.2 Visual Inspection**

This matrix should be diagonal, or at least sparse. The sample covariance matrix based on estimated idiosyncratic returns does not satisfy these requirements. The sample covariance matrix $\hat{\Omega}_\epsilon = \frac{1}{T} \sum_{t=1}^T \hat{\mathbf{\epsilon}}_t \hat{\mathbf{\epsilon}}_t^T$ is not sparse (unless $N \ll T$). One possibility could be all the off-diagonal terms to zero, thus effectively amounts to a radical shrinkage of the idiosyncratic correlations. This step, however, is not warranted. As a sanity check, we recommend performing a visual inspection of the idiosyncratic covariance matrix. Oftentimes, there are strong patterns that can be interpreted as factors. For example, there are blocks of stocks that are highly correlated (e.g., some Chinese securities are listed in both Mainland China and in Hong Kong (H shares)). These securities have highly correlated but not identical returns, and the correlations will show up in the idiosyncratic covariance matrix. In such a case, rather than assuming that $\hat{\Omega}_\epsilon$ is sparse, it is more appropriate to add a “local factor” to the model.

**6.4.3 Short-Term Idio Update**

Idiosyncratic returns, like factor returns, are subject to sudden changes in volatility that are not captured well by exponential weighting with long half-lives $T$.
A very responsive daily return model has $T \approx 126$ trading days, and the shocks may occur over 10 trading days. As a remedy, we reuse the STVU mechanism of Section 6.3.3. We estimate the idiosyncratic volatilities using the same formula, although the technique is applicable to other asset classes. Stocks are likely to receive large shocks in proximity of earnings, either because new information is released before or on earnings date, or because such information becomes fully priced in the following days. We use a two-tier horizon during which earnings information is received. Define the function as
$$ s_{i,t} = \begin{cases} 1 - |t-T_{earn,i}|/\tau_{earn} & \text{if } |t-T_{earn,i}| \le \tau_{earn} \\ 0 & \text{otherwise} \end{cases} $$
$s_{i,t}$ ranges from zero to one, when $t$ is within $T_{earn}$ number of days from the earnings date $T_{earn,i}$ (see Model 6.8), but restricting ourselves to positive values. We then modify the correction term in the STVU mechanism. Instead of the simple fact that the correlation matrix is approximately diagonal. We restrict our attention to the linear approximation; the corrective term is
$$ \hat{\sigma}_{\epsilon,it}^{2,c} = \hat{\sigma}_{\epsilon,it}^2 \exp \left( \sum_{k=0}^{K_0-1} w_k \frac{\sum_{j \in S_k} \alpha_{i,j} (\hat{\epsilon}_{i, t-j}^2 - \hat{\sigma}_{\epsilon, i, t-j}^2)}{\sum_{j \in S_k} \alpha_{i,j} \hat{\sigma}_{\epsilon, i, t-j}^2} \right) $$
and applies only to the assets affected by earnings.
$$ \hat{\sigma}_{\epsilon,it}^{2,c} \leftarrow [1 - \alpha_{i,t} (1-e^{-1/\tau})] \hat{\sigma}_{\epsilon,it}^2 $$

**6.4.4 Off-Diagonal Clustering**

Finally, we need to identify those assets whose idiosyncratic returns are highly correlated. Two instances are important. The first one is the case of different securities that refer to the same underlying asset. For example, some stocks are listed on different share classes, for example, Berkshire Hathaway trades under BRK.A and BRK.B, with different fractional values. The liquidity of the two securities may be different. The second one is when the correlation between securities in a factor model still depends on the nature of the trading strategy employing the model itself. If the strategy intends to exploit the temporary small mispricing between two assets, then we should include both assets. If instead we only intend to invest in the company based on fundamental information, then we should only include a security representative of the underlying asset, typically the one with the highest liquidity. The second one is when the correlation between stocks whose comovements are not described by factors. In order to be identifiable, factors must be pervasive. A factor influencing only a handful of assets is not a factor, and cannot be identified in a large cross-section of assets. The dependency among these stocks is still available in the correlation between their idiosyncratic returns. We can estimate these correlations. We can transform the correlation elements by applying a simple sparsifying operator.
The optimal threshold is $h_c = \rho_c \sqrt{K \log K / T}$ (Cai et al., 2011), for some positive constant $K$. In practice, however, it is more instructive to explore the clusters emerging for different values of the threshold $\lambda$. For some value of $\lambda$, the clusters are (a) stable, in the sense that they do not change much for perturbed values of $\lambda$, and (b) interpretable. They are composed of “similar” stocks, such as stocks in the same sector, or stocks with similar size and sometimes style factor loadings as well.[8] It is important to check that, for every level of the threshold, the correlation matrix is symmetric positive-definite (and well-conditioned). For example, we use the residual returns from a commercial factor model (Axioma US4 Short Horizon, 2015-06-08). We compute the equal-weighted correlations for consecutive constituents of the Russell 3000 index, and threshold their absolute values at 0.15. Figure 6.1 shows the resulting clusters. The number of stocks is small: less than 3% out of a total of nearly 3,000 stocks. The plot is quite sparse. The plot also includes Las Vegas Sands, Peabody Energy and Alpha Natural Resources, and a mining cluster composed of HL, NEM, CDE, PCLD.

[Image: Figure 6.1: Clusters for idiosyncratic matrix.]

**TABLE 6.1**
Ticker and company names of cluster components in Figure 6.1

| Ticker Name | Ticker Name |
| :---------- | :---------- |
| AAPL APPLE INC | GOOG ALPHABET INC CL C |
| ...         | ...         |

**6.4.5 Idiosyncratic Covariance Matrix Shrinkage**

Analogous to the factor covariance shrinkage of Section 6.3.1, we recommend shrinking the idiosyncratic variances toward the identity matrix.
$$ \Omega_{\epsilon,shrink}(\rho) = (1-\rho) \hat{\Omega}_\epsilon + \rho \frac{\text{trace}(\hat{\Omega}_\epsilon)}{m} \mathbf{I}_m $$
For a diagonal idiosyncratic covariance matrix, this is analogous to shrinking the idiosyncratic variances toward the empirical average of the variances.

---

**6.5 Winsorization of Returns**

The issue of outlier detection, if not central, is at least very important both for risk modeling and alpha research. There are many examples of outliers in return data, each one of them responsible for ruining the career of a finance researcher. Before proposing some remedial measures to improve the research process and save a few careers, let us discuss where they come from, and the impact that they may have. The sources are several and diverse. First, the data provider may have made a mistake. Secondly, the data are genuinely very large. This is the case of liquidity-driven volatility. Third, good researchers spend a large proportion of their time evaluating and comparing data and questioning providers on the data collection methodology and their applicability. The sources of error are many. In the worst case, prices are not comparable across assets or across time. In the best case, prices are not comparable across assets. In the best case, prices are not comparable across assets. In the best case, the ultimate source of the returns may be a broker located in a farming village in New Zealand, and even more unlikely instances. Bad providers are the perfect breeding ground for outliers. Second, authentic outliers do exist. Instances are:
*   There are rare stray transactions for a liquid stock.[9]
*   Very illiquid stocks exhibit higher volatility, and occasionally large returns.
*   Stocks in the process of being delisted, or entering/exiting bankruptcy, usually trading over-the-counter (OTC) also exhibit very large returns.
*   The most pervasive large jumps reflecting new information in the market, say in earnings and forward guidance, announcement of market entry by a competitor, merger announcements and news of merger break, accidents or hiring new and unforeseen liabilities, and macroeconomic drivers having a large impact on the security’s price.
Of these instances, the first one is related to improving price data collection, i.e., by not only taking the last bid price of a 5-minute interval, but also considering the entire price trace. The second and third can be avoided by choosing the estimation and investment universe carefully. Microcap stocks that are very illiquid should be excluded; the last class of outliers, however large they may be, should not be excluded from the analysis. Instead, the model should be made robust to them. Make the output of the analysis much less reliable. If winsorizing a large absolute return, we affect the estimated factor returns from the cross-sectional regression. A factor return is the return of its mimicking portfolio, and by winsorizing returns in the cross-sectional regression, we effectively cap the portfolio return from that period. The total portfolio return is inevitably capped based on historical returns. This affects the evaluation of the factor returns, as well as of the idiosyncratic returns. Two easy qualitative recommendations that follow from all of this are:
1.  Whatever winsorization method you use, make sure to report all the instances of winsorized data in backtest or in production, and examine them one by one.
2.  Make sure that your investment universe comprises only liquid, tradable assets.
For the winsorization process, use a robust outlier detection strategy. There is no ideal and completely justified method. A method that works well enough is to compute at the security level, the robust z-score of return:
$$ d_{it} = \frac{|\log(1+r_{it}) - \text{median}_s \log(1+r_{is})|}{\text{median}_s |\log(1+r_{is}) - \text{median}_k \log(1+r_{ik})|} $$
and to winsorize returns exceeding a threshold $d_{max}$. The threshold depends on asset class, region, and other attributes, and is set by trial-and-error between 5 and 10.

---

**6.6 Advanced Model Topics**

**6.6.1 Linking Models**

In some applications, you will have an investment universe consisting of securities belonging to different regions. For large hedge funds, for example, most of their assets are in North America, Europe, and Asia. Linking models allows the investment of securities belonging to different asset classes. An example is U.S. equity and corporate credit, but the possibilities are endless. Since the mathematical treatment that will follow is somewhat dry and arduous, it is perhaps worth asking the question: why are we here? The first one is the case of integrated factor models. Secondly, what then are the benefits of linking models? The first one is the most obvious (or wide) risk management. We need an accurate a measure of firm-wide volatility as possible. Being wrong by 10% in errors or defect in the firm value is either damaging returns because we have deployed more assets than we should have, or because we have deployed too few, thus missing a good opportunity. A second one is that of a large drawdown. A drawdown that is caused by a failure of our risk model is bad in itself—of course!—but then it may turn into a positive-definite matrix. We need a factor model for alpha research and portfolio optimization. In my personal experience, linking models is the last part of your problem when it is your only one.
In principle, it is possible to model “everything”: across asset classes and geographies. Figure 6.2 shows simplified instances of asset and geography models. In practice, the modeler is faced with three options. The boundaries between them are not perfectly demarcated, the description will target only the idealized cases.

[Image: Figure 6.2: Left: credit-equity-linked factor covariance matrix. Right: country-linked factor covariance matrix. Diagrams show blocks for Credit Factor Cov, Equity Factor Cov, US Equity Factor Cov, EU Equity Factor Cov, etc.]

1.  You may want to jointly model all the assets. This is not any different approach, see Burns et al. (1998). Regarding returns, one possible approach is to estimate a single factor model that includes all assets. The availability of data for all local models, on the same dates will ensure that the factor covariance matrix is positive-definite. It is also possible to estimate factor returns (for, e.g., performance attribution), an increase in estimation error may be the drawback. The factor model is also likely to be very large, and tracking portfolios have potentially high turnover as days when the estimation universe changes due to missing returns for certain local models. Yet another approach is not to rely on the factor model when data are not available, and instead to impute returns for these assets; we won’t cover it here.[10]
    Regarding missing returns, it is also possible (although slightly unlikely) to estimate factor returns on all dates, and then to use these factor returns to estimate the factor covariance matrix using only complete observations. The factor covariance matrix will not necessarily be positive-definite, but it may turn into a positive-definite matrix by shrinking the off-diagonal conservation matrix. This is the approach taken by ANCIENT LOKKE.[11]
2.  Separate Modeling. From the previous item, we learn that integrated modeling can be both powerful and dangerous. In the next approach, we seek to keep most of the benefits of the integrated approach, while keeping the models separate. We estimate local models as a starting point. We denote the local factor covariance models as $\hat{\Omega}_{f,i}^{(L)}$. In addition, we have an integrated model, where factors are “integrated”, i.e., estimated using the full universe of assets. Let $\mathbf{B}_{i \cdot}^{(I)}$ be the estimated factor loadings for the universe of market $i$ and market $j$. The idiosyncratic covariance between securities in market $i$ and market $j$ is block-diagonal.
    $$ \hat{\Omega}_{f}^{(I)} = \begin{pmatrix} \hat{\Omega}_{f,1}^{(I)} & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & \hat{\Omega}_{f,P}^{(I)} \end{pmatrix} $$
    Here is the form of the integrated model:
    (Link)
    from jointly modeling assets belonging to different sectors in equity models. You could include country factors in the models, and do businesses.
    You may develop separate models (which we call local models) for, say, assets belonging to different geographies and then combine the models in a second stage. The distinguishing feature of this approach is that we want to make sure that the integrated model, when restricted to each local market, is identical to the local model.
    You may model the assets jointly in a first stage, and in a second-stage model the residual returns obtained from the first stage in separate models.
    We discuss the benefits.
    1.  Integrated Modeling. This approach is conceptually straightforward, and has the advantage of giving the model the greatest deal of flexibility when modeling the relationships between securities. For example, in an equity-corporate credit model we may want to create a factor describing Investment Grade versus High-Yield classification. This factor could affect all bonds and stocks. However, the resulting IC loadings from a specific sector/sub-sector are best incorporated in an integrated model. Moreover, if the model is used for trading, alpha and factor-mimicking portfolios should be developed and tested for the combined investment universe. There are some caveats. First, the estimation universe may be very large, especially if some zones, when securities are in multiple countries. Another source of complication comes from the misaligned trading calendars of different markets and asset classes. Trading holidays differ by country, so we have to deal with asynchronous observations, which can result in stale prices. This may result in factor returns for certain factors on certain days. These, in turn, can result in non-positive-definite factor covariance matrices. There are strategies to address some of these issues, which we outline below.
    2.  One way to address asynchronous data is by aggregating returns over multiple days, ranging from two to five. This way, returns become largely overlapping. The number of observations within a time window is reduced accordingly, and the time scaling of volatility, approximately as $\sqrt{T}$.
    3.  Asynchronicity induces cross-autocorrelation in factor returns. One way to partially address the problem is by applying the Newey-West estimator to the covariance matrix. Other, more advanced methods, for example by explicitly modeling returns in a state-space framework. Treatment falls beyond the scope of this book. For an instance of this
    $$ \mathbf{r}_t^{(I)} = \begin{pmatrix} \mathbf{B}^{(1)} & 0 & \dots & 0 \\ 0 & \mathbf{B}^{(2)} & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \mathbf{B}^{(P)} \end{pmatrix} \begin{pmatrix} \mathbf{f}_t^{(1)} \\ \mathbf{f}_t^{(2)} \\ \vdots \\ \mathbf{f}_t^{(P)} \end{pmatrix} + \mathbf{\epsilon}_t^{(I)} $$
    The integrated model’s covariance matrix is
    $$ \hat{\Omega}_f^{(I)} = \mathbf{B}^{(I)} (\mathbf{B}^{(I)})^T + \hat{\Omega}_\epsilon^{(I)} $$
    Let us go back to our challenge. The local models are valuable, because they are estimated using all the data, e.g., all the market data, and are market-specific, not factor-specific. We want to quantify their joint risk; that is the point of having a linked model. The idea is to develop a “minimal rotation” $C^{(I)}$ that transforms $\hat{\Omega}_f^{(I)}$ into a matrix where the local factor covariance matrices are preserved, and the cross-market covariances are estimated using the integrated model. It will be sufficient to require the rotation to be block-diagonal
    $$ C^{(I)} := \text{diag}(C^{(1)}, \dots, C^{(P)}) $$
    (this is the clever bit)
    $$ C^{(I)} \hat{\Omega}_f^{(I)} (C^{(I)})^T = \hat{\Omega}_f^{(L)} $$
    Note that $C^{(I)}$ should be close to the identity, because we expect $\hat{\Omega}_{f,ii}^{(L)}$ and $\hat{\Omega}_{f,ii}^{(I)}$ to be very similar. Let us write the rotation down (the “IR” superscript stands for “Integrated Rotated”):
    $$ \hat{\Omega}_{f}^{(IR)} = \begin{pmatrix} \hat{\Omega}_{f,11}^{(L)} & C_{12}^{(L)} & \dots & C_{1P}^{(L)} \\ C_{21}^{(L)} & \hat{\Omega}_{f,22}^{(L)} & \dots & C_{2P}^{(L)} \\ \vdots & \vdots & \ddots & \vdots \\ C_{P1}^{(L)} & C_{P2}^{(L)} & \dots & \hat{\Omega}_{f,PP}^{(L)} \end{pmatrix} $$
    Here,
    $$ C_{ij}^{(L)} := (C^{(I)} \hat{\Omega}_f^{(I)} (C^{(I)})^T)_{ij} $$
    Note that we are not transforming $\hat{\Omega}_f^{(I)}$ by post-multiplying it by $C^{(I)}$. Doing so would change the local market loadings, and as a result we would not preserve the local market model. In other words, we are not actually rotating the entire model, we are only perturbing the factor covariance matrix by a small amount, for alignment purposes. The error induced in the $\mathbf{V}_{i,j}^{(L)}$ is small and should be tolerable. Finally, it is of interest to note that the resulting covariance matrix is block-diagonal, so that, by construction (see Exercise 6.3), the final covariance matrix has the form (the “IRS” superscript stands for “Integrated Rotated Shrunken”):
    $$ \hat{\Omega}_f^{(IRS)} = \begin{pmatrix} \hat{\Omega}_{f,11}^{(L)} & \rho \hat{\mathbf{V}}_{1,2}^{(L)} & \dots & \rho \hat{\mathbf{V}}_{1,P}^{(L)} \\ \rho \hat{\mathbf{V}}_{2,1}^{(L)} & \hat{\Omega}_{f,22}^{(L)} & \dots & \rho \hat{\mathbf{V}}_{2,P}^{(L)} \\ \vdots & \vdots & \ddots & \vdots \\ \rho \hat{\mathbf{V}}_{P,1}^{(L)} & \rho \hat{\mathbf{V}}_{P,2}^{(L)} & \dots & \hat{\Omega}_{f,PP}^{(L)} \end{pmatrix} $$
3.  Multi-stage Modeling. There are use cases in which we would like to have both “global” and “local” factors. The global factors describe the co-movements among securities across all the local markets. The local factors are instead affecting only the securities in their market. The appeal to this approach is that it makes the local factors independent of each other. When we describe a way to obtain such a model. As a premise, we need an integrated model $\hat{\Omega}_f^{(I)}$, or an integrated and rotated model $\hat{\Omega}_f^{(IR)}$. In addition, we need global characteristics $\mathbf{B}^{(G)}$ that describe, for “country” factors (all loadings equal to one):
    *   a global market factor;
    *   global style factors, namely momentum, volatility, and value.
    In the first step, we perform cross-sectional regressions on the integrated universe:
    $$ \begin{pmatrix} \mathbf{f}_t^{(1)} \\ \mathbf{f}_t^{(2)} \\ \vdots \\ \mathbf{f}_t^{(P)} \end{pmatrix} = \mathbf{B}^{(G)} \mathbf{g}_t $$
    and so we obtain global factor returns $\hat{\mathbf{g}}_t$. We arrange the global factor returns into a matrix $\mathbf{G} \in \mathbb{R}^{K_G \times T}$, whose columns are $\hat{\mathbf{g}}_t$. In the second step, we regress the local factors against the global factors.
    $$ \hat{\mathbf{f}}_t^{(i)} = C^{(i)} \mathbf{g}_t + \mathbf{f}_t^{(i), \perp} $$
    The solution to this regression is given by Equation (6.4).
    $$ C^{(i)} = (\mathbf{G}^T \mathbf{G})^{-1} \mathbf{G}^T \mathbf{F}^{(i)} $$
    In the third step, we replace these formulas in the Integrated Factor Model Equation (6.10), to get
    $$ \mathbf{r}_t^{(i)} = \mathbf{B}^{(i)} C^{(i)} \mathbf{g}_t + \mathbf{B}^{(i)} \mathbf{f}_t^{(i), \perp} + \mathbf{\epsilon}_t^{(i)} $$
    We must check that the returns vectors $\mathbf{f}_t^{(i), \perp}$ are pairwise approximately uncorrelated. In this case, the factor covariance matrix is
    $$ \text{diag}(\text{cov}(\mathbf{g}_t), \text{cov}(\mathbf{f}_t^{(1), \perp}), \dots, \text{cov}(\mathbf{f}_t^{(P), \perp})) $$
    If that is not the case, then there are three options. First, add global factors while keeping $m_G$ constant. Second, add global factors in order to achieve our goal. Third, give up finance altogether.

**6.6.2 Currency Hedging**

In a multi-country factor model, the return of an asset is usually expressed in a different currency than the one in which the asset is traded. Consider the case of a U.S. investor holding a security denominated in Euros. We define the return of the asset in Euros (the local currency) and the return of the Euro in U.S. dollars (the reference currency is the security is traded [the Euro], also called the base currency, and sell dollars, the quote currency. The direct exchange rate is the Dollar amount needed to buy one Euro. More details on usage when referring to a currency pair are in Section 1.6.2). The return of the Euro in U.S. dollars is the direct exchange rate in period $t$ ($P_{EURUSD,t}$) divided by the direct exchange rate in period $t-1$ ($P_{EURUSD,t-1}$). The indirect exchange rate is the exchange rate of the reversed pair, and is equal to the reciprocal of the direct exchange rate. The return is defined as the return received by holding one base currency unit, expressed in quote currency.
$(1+r_{EURUSD,t}) = P_{EURUSD,t} / P_{EURUSD,t-1}$.
We define the currency return $r_{EUR,t}$ when the base and quote currencies are the same to be zero.
Let us analyze first the realized return of holding EUR in a simple transaction, in which we buy and sell the currency on consecutive days. We denote by $r_{EUR,t}^{USD}$ the risk-free interest rate in the interval between the two transaction spots for the two currencies.
On day 0, we borrow 1 EUR and purchase $(1+r_{EUR,t}^{USD}) P_{EURUSD,(0)}$ EUR.
On day 1, the EUR holding is worth
$$ (1+r_{EUR,t}^{USD}) P_{EURUSD,(1)} / P_{EURUSD,(0)} $$
We buy back Dollars at the price $P_{EURUSD,(1)}$. The Dollar amount we are left with is
$$ (1+r_{EUR,t}^{USD}) P_{EURUSD,(1)} / P_{EURUSD,(0)} $$
$$ = (1+r_{EUR,t}^{USD}) (1+r_{EURUSD,t}) $$
We then pay our USD loan for an amount $-1 \cdot r_{USD,t}$. We are left with
$$ (1+r_{EUR,t}^{USD}) (1+r_{EURUSD,t}) - 1 - r_{USD,t} \approx r_{EUR,t}^{USD} + r_{EURUSD,t} - r_{USD,t} $$
The return $r_{EUR,t}^{USD}$ is the currency return adjusted by the difference in risk-free rates. This result is intuitive. Instead of holding the EUR in a cash account, we invest it in a risk-free asset. With local currency $r_{LC,t}$ (EUR) equal to $r_{EUR,t}^{USD}$, following the same calculations, the return is
$$ r_t^{USD} = r_{LC,t} + r_{EURUSD,t} - r_{USD,t} $$
$$ r_t^{LC} := r_t^{EUR} $$
The return is the sum of two components: the local excess return and the adjusted currency return.
There is yet another identity, which links the currency returns of three (or more) currencies. A no-arbitrage argument along the same lines as above yields $r_{A,t}^C = r_{A,t}^B + r_{B,t}^C$, from which the identity holds:
(Heston, 1993) $r_{A,t}^C - r_{A,t}^B = r_{B,t}^C$
In matrix form, we write the identity as
(6.14)
$$ \mathbf{r}_t^C = (I_N - A^{(B)}) \mathbf{r}_t^B $$
(6.15)
$$ [A^{(B)}]_{m,n} = \begin{cases} 1 & \text{if } n \ne k \\ 0 & \text{if } n=k \end{cases} $$
Now we consider the problem of changing numeraires. For example, we want to change the numeraire from USD to GBP.
(6.16)
$$ r_{t,EURGBP}^{USD} = r_{t,EUR}^{USD} + r_{t,USDGBP}^{USD} $$
Let us say that our factor model contains securities traded in $K$ currencies. The assets total returns in base currency $\mathbf{r}_t^B$ can be decomposed into the sum of a local excess return and an exchange rate return.
$$ \mathbf{r}_t = \mathbf{B} \mathbf{f}_t + \mathbf{\epsilon}_t + C \mathbf{r}_{FX,t}^B $$
(local factor structure) (currency factor structure)
The elements of the matrix $C \in \mathbb{R}^{N \times K}$ take 0 or 1 values, with $C_{i,j}=1$ if asset $i$ has reference currency $j$ and 0 otherwise. We reuse from currency $k$ to currency $l$ by way of transforming the currency returns using Equation (6.15).
$$ \mathbf{r}_t = \mathbf{B} \mathbf{f}_t + \mathbf{\epsilon}_t + C (I_K - A^{(k)}) \mathbf{r}_{FX,t}^k $$
We close this section with several comments related to modeling extensions and practical implementation.
We have ignored the question of modeling the joint distribution of the spot currency returns $\mathbf{r}_{FX,t}^k$. One natural avenue is to model those using a factor model, either fundamental or statistical, so that we can express $\mathbf{r}_{FX,t}^k = \mathbf{B}_{FX} \mathbf{f}_{FX,t} + \mathbf{\epsilon}_{FX,t}$. We need to model the relationship only with respect to the spot currency.
Finally, we have ignored the covariances between $\mathbf{f}$ and $\mathbf{\epsilon}$. The complete model is
$$ \mathbf{R}_t = \begin{bmatrix} \mathbf{B} & C \mathbf{B}_{FX} & C \\ \mathbf{0} & \mathbf{B}_{FX} & I_K \\ \mathbf{0} & \mathbf{0} & I_K \end{bmatrix} \begin{pmatrix} \mathbf{f}_t \\ \mathbf{f}_{FX,t} \\ \mathbf{g}_t \end{pmatrix} + \mathbf{\epsilon}_t $$
Currency risk depends heavily on the contractual arrangement of the investment vehicle. For example, an investment firm may have fixed capital housed in a different country, and trade using only this capital as collateral. The net exposure is fixed. In this case the foreign currency exposure is given by the capital level, which is usually hedged by currency forward contracts.

---

**6.7 A Tour of Factors**

This chapter would not be complete without at least a cursory description of fundamental factors. Because factors explain cross-sectional returns, the factor literature is constantly in the financial literature exploring return anomalies and extensions to the CAPM or the standard Fama-French three-factor model. The literature on factor anomalies is vast. There are several reviews and introducing books on this subject (see Bali et al. (2016), Cochrane (2005), Harvey et al. (2016)). This section is a “Factor Zoo” to denote the large set of published factor anomalies introduced since 2011, a set that has greatly expanded since then. Whether these factors represent true, traded anomalies is still being debated. Harvey and Liu (2019) develop statistical methods and apply them to the factor zoo, concluding that most factor anomalies are false positives (Harvey and Liu, 2019, 2020b). On the other side, Jensen et al. (2019) argues that most anomalies are real. Chen (2020a) reconciles the two papers.
Some papers reviewing the large number of factors and attempting to classify them into a smaller set of clusters are Jacobs (2015), Frazzini et al. (2018), Feng et al. (2020), Hou et al. (2015). The list below is not based on these classifications only, but takes into account the factors included in commercial models.
*   Market. By far the most pervasive factor in the data, it is usually represented as the market capitalization-weighted return of the asset universe (United States). In the first case, the interpretation is that any return is dominated by the market factor. In the second case, it is left to other factors to capture the systematic risk.
*   Size. Small companies and countries are usually represented as a common, integrated (0/1) variable summing to one for each asset, although non-integer loadings are possible. We consider these factors as homogeneous not only because information is coded in the loadings, but also because the same interpretation of the factor returns holds.
*   Value. Value stocks have outperformed growth stocks, but not consistently. Aside from the papers by Heston and Rowenhorst cited above, see also Caraglia et al. (2009), Grinblatt and Moskowitz (2004), Daniel and Titman (1997), Fama and French (1992), Pástor et al. (2008).
*   Momentum. Stocks that have outperformed (underperformed) their peers over the 3-12 months previous to a given date often outperform (underperform) their peers in the future. Jegadeesh and Titman (1993) documented this anomaly in the academic literature. They review the evidence of Litterman, Jegadeesh and Titman (1999), Asness (1994), Moskowitz and Grinblatt (1999), and West (2003).
*   Reversal. Stocks that have outperformed in the recent past typically one month or less) underperform. More recent outperformance is more predictive, so the effect is stronger for past-week outperformers than for past-month outperformers. See Jegadeesh (1990), Lehmann (1990).
*   Fundamental Valuation. There are many characteristics descriptive of over/underpricing of firms, based on fundamental data. An early example is Book-to-Price (Fama and French, 1992). Other characteristics are Earnings-to-Price (Basu, 1977), Dividend Yield (Litzenberger and Ramaswamy, 1979), and various growth measures, which often employ metrics that are customized by the sectors in which the firms operate.
*   Low Beta/Low Vol. High-beta stocks have lower risk-adjusted performance than low-beta stocks (Black, Jensen, Scholes, 1972), and the anomaly is sometimes called the Betting Against Beta factor. Similarly, high-volatility stocks have lower risk-adjusted performance than low-volatility stocks (Black, 1972; Haugen and Heins, 1975). The two effects are related, since the beta of a stock to a benchmark is $\beta_i = \rho_{i,M} \sigma_i / \sigma_M$. In a portfolio that is long low-beta (low-vol) stocks and short high-beta (high-vol) stocks, Frazzini and Pedersen (2014), Blitz et al. (2019), Treat (2017).
*   Liquidity. Stocks that are more illiquid in recent periods outperform more liquid stocks (Amihud and Mendelson, 1986; Amihud, 2002). A characteristic that describes liquidity is Amihud’s measure:
    $$ ILLIQ_{i,d} = \frac{1}{T} \sum_{t=1}^T \frac{|R_{i,t,d}|}{VOLD_{i,t,d}} $$
    where $ILLIQ_{i,d}$ is the daily return of stock $i$ on day $d$ and $VOLD_{i,d}$ is the dollar volume of stock $i$ on the same day. A possible interpretation of the role that illiquidity plays in the returns of stocks (and bonds; see Chen et al. (2007)) is that investors are compensated for bearing illiquidity risk by receiving a higher return during periods of market stress. These are the short-lived intervals in which market participants are forced to liquidate their bonds due to their own idiosyncratic liquidity needs. These actions are due to issues that are not related to the fundamental value of the asset, see Mitchell et al. (2002), Brunnermeier and Pedersen (2009).
*   Crowding. Forced liquidations are more likely to occur when certain positions are crowded, i.e., when many firms with similar trading characteristics hold them. When a position (or, more broadly, a portfolio) is crowded, its liquidation by one of the holders causes an adverse price change that negatively affects PnL of the other holders. If this effect is large enough, then other holders may also liquidate their positions, exacerbating the overlap among portfolios and positions, we can quantify the size of a negative return caused by the liquidation. Since shorting stocks is usually performed by informed traders, short interest data is a source of crowding information on the short side. On the long side, Active Manager holdings are reported at the security and holder level by institutional investors (SEC, 1934).
*   Size. Small companies are outperforming large ones. First introduced by Banz (1981), the anomaly refers to the outperformance of (next to) the smallest market capitalization decile firms. Smaller firms have higher leverage and higher earnings variation. There is also a behavioral explanation: small firm stocks are “lottery tickets” (Barberis and Huang, 2008); this results in overpricing of the small-cap stocks. The mispricing cannot be corrected by informed investors, either because they have a mandate not to short stocks, or because shorting shares is expensive due to high borrowing costs.
I close this section with some subjective remarks on the factor literature.
First, this classification, while broad, is incomplete. For example, it has not included several well-known factors, like Quality or Investment. It has not included several well-known factors, like Quality or Investment. Researchers publish new factors, sometimes based on interactions among existing factors, sometimes based on increasingly elaborate characteristics. It is unclear that these factors describe true pricing anomalies or are just variants of factors listed above.
This leads to the second point. The explanatory power of a factor is tested against an elementary model, the Fama-French three-factor model (Fama and French (1993)) or one of its refinements, such as the five-factor model (Fama and French (2015)). There is an implicit assumption that the factors are orthogonal to each other. When a new factor is proposed, its returns are often highly correlated to existing factors. For example, in Cochrane (2011) a new self-help manual is proposed. It is explained in simple terms to employ it. It is profitable from sorts and ignores turnover, borrow costs for short positions, and transaction costs.
Lastly, authors may have tried many variants of a characteristic in order to obtain a positive result. These results are ex-post means much higher than the true means. This is termed “p-hacking” or “data snooping”.
The central question is whether the published anomalies are tradable, profitable factors. My subjective answer is nuanced. First, because of the considerations above, many of the published anomalies are non-existent in the first place, non-tradable, or non-profitable after transaction costs (McLean and Pontiff, 2016). Second, there is a seasonal misapprehension of value by investors. Oftentimes, a new phenomenon is successfully signal-researched before retiring. In their own words, “my top twenty lines were the vast majority of my PnL producers. Each one of them could be described in four lines: the details that matter”, to which he replied (I believe) “my signal design portfolio construction and risk management are all twenty lines long, and I need additional real-time tuning of parameters. All these qualifications leave room for some factors to be tradable and profitable.

**The Takeaways**

1.  Fundamental factor models rely on asset returns and characteristic data to generate factor and idiosyncratic return estimates.
2.  Strengths: Good performance, interpretability, connections to academic models, and utility in alpha research.
3.  There are six major steps in model identification:
    *   Data ingestion and integrity checks.
    *   Selection of a universe of tradable, liquid assets for estimation.
    *   Winsorization of returns to handle outliers.
    *   Generation of factor loadings from asset characteristics.
    *   Cross-sectional regression of factor returns.
    *   Covariance estimation of factors and idiosyncratic returns.
4.  Address rank-deficiency by using weighted least squares improves model stability.
5.  Factor covariance matrix requires shrinkage or dynamic conditional correlation adjustments to account for estimation error and autocorrelation.
6.  Idiosyncratic covariance matrix estimation benefits from short-term updates and clustering to capture residual correlations.
7.  Linking models improves risk management, enables multi-asset class tradability across different geographic regions.
8.  Linking different factor models allows integration across regions or asset classes.

**Notes**

1.  [1] As of October 2012, in North American markets, it is higher on days when indices are recontaminated.
2.  [2] The inputs are explained in Section 4.1, we denote $\mathbf{B}_t$ the loadings available at the end of period $t-1$.
3.  [3] See Equation 2.4.2.
4.  [4] See Section 2.4.2.
5.  [5] We could characterize more rigorously this within-cluster similarity as a distance among factor loadings, and given this measure, we could pursue a systematic thresholding procedure, but it would fall beyond the scope of the book.
6.  [6] If I can self-indulge in a personal recollection: I was working for Enron in the summer of 2001. One day, at the trading meeting, Enron’s stock price dropped by 80% after news that Enron had $8B of debt. Someone on Enron’s trading floor were openly wondering “Should we short Enron now?” The stock fell back to mid-80s the day after. Lesson: if traders want to short themselves, then it’s a likely outlier of some kind.
7.  [7] As a justification for going back to our original subject, I note that the problem of estimating a population mean when data is high, and the return on the effort in this case is comparatively low.
8.  [8] This is the phrase used by medieval cartographers to denote the borders beyond which lay unchartered territories.
9.  [9] The returns corresponding to the local factor returns $C_j^{(L)}$ are homoskedastic. This is generally not the case. However, it is possible to further refine the model in order to allow for heteroskedasticity. Turtles all the way down.
10. [10] The currency codes are identified by three letters. The most common are USD (US Dollar), EUR (Euro), JPY (Japanese Yen), GBP (UK Pounds), AUD (Australian Dollars), CAD (Canadian Dollars), CNY (Yuan Renminbi), JPY (Yen).
   

Okay, here is the Markdown compilation for Chapter 7.
     Everand

