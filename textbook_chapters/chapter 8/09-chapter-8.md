**Chapter 8**
**Evaluating Excess Returns**

**The Questions**

1.  How does revising historical data impact alpha estimation and backtesting in factor models?
2.  What are the best practices we should follow in our backtests?
3.  What are the two main backtesting frameworks and what are their drawbacks?
4.  What approach could address these limitations?

The task of estimating factor models and testing alphas for systematic strategies usually involves reusing the same historical data. This is a major statistical conundrum. One of the defining features of the past 20 years has been the unprecedented number of new datasets and their broad dissemination. Investment firms spend large budgets of tens of millions of dollars allocated to the purchase of market and alternative data, and to the bespoke collection of data (e.g., via web scraping). And yet, the characteristics of the data are often poorly understood. The time series of data—the building blocks of our models and simulations—are often short, the number of times—of the order of a day or longer—do not necessarily employ tick-by-tick data. If we record prices for a broad local investment universe at 1-minute intervals, we collect 60 million numeric data points per year, including a security identifier and a timestamp. The required storage for this data is modest; the quality is not. It is increasingly common to deal with deep data. Deep data means that the past is not easy to produce a simulated version of the past that provably reproduces all of its features. Not deep, in the sense that we do not live in a time series. The world is much richer. The real world outside of finance is sprawling, breathtaking and awe-inspiring, and models are crude approximations. But even within finance, the introduction of new technology, of new market microstructure designs, of new regulations, and the ongoing collective learning process of all market participants make the investing world of five years ago very different from today’s. The fact that we have more data on our hard drives poses a major challenge to the modeler. Financial practitioners do not have a shared protocol for experimental analysis. Even if we had one, it is far from obvious that it is the correct one. Well-established disciplines like medicine and psychology have shared experimental practices accompanied by experimental design, and yet they have undergone a crisis of confidence when their peers found that some of their results were not replicable (Greenland, 2017; Open Science Collaboration, 2015).

This poses a few challenges for us modelers. We have a very large number of signal types, which themselves depend on continuous tuning parameters, and we only have a limited history. This is similar to the situation faced by biostatisticians, who deal with tens of thousands of simultaneous tests in the form of responses from a DNA microarray (Dudoit et al., 2003; Storey et al., 2004). The details are quite different. The response variable for a DNA microarray is usually continuous. For financial signals, returns are autocorrelated or weakly correlated. In quantitative finance, the response variable (be it return or Sharpe), is continuous, and signal correlation plays a decisive role.

This chapter has four sections. First, we list some basic best practices for data preparation and usage. Second, we describe some common backtesting practices and criticize them. The third section is entirely devoted to describing a new backtesting protocol, which is based on recent advances in statistical learning theory. We derive simple uniform probabilistic bounds on the Sharpe Ratio of a large set of strategies. The last section applies the theory developed so far to simulated and historical data.

---

**8.1 Backtesting Best Practices**

We review in this section practices for backtesting. They do not originate from some comprehensive Theory of Everything, but are born fully formed from the mind of Zeus. It is an ever-incomplete, occasionally shallow body of knowledge that has formed by experimentation. Some references covering these practices are Wang et al. (2014); Arnosti et al. (2017); López de Prado (2018).

**8.1.1 Data Sourcing**

High-quality data are essential to backtesting, and the search for better data is a never-ending task for a researcher. There are several broad areas of concern. The first one is data sourcing. There are multiple vendors offering similar data. When comparing them, ask the following questions:
*   Definition and interpretation. Perhaps the first and most important question is not only in data sourcing, but in quantitative finance in general, what do the data mean? Do we know the exact definition of the data? What are their physical units? If the dataset is money-related, it should be unambiguous what is the reference currency (or currencies, for exchange rates). If the data is flow-related (i.e., measuring units over time), the time unit should be defined. A common source of error is deciding between annual or annualized numbers.
*   Provenance. Where are the data coming from? Does the vendor collect the data themselves (e.g., via web scraping, or internet traffic)? Does the vendor buy the data from another vendor, and then act as a data integrator and the client? In the former case, what is the data collection criterion? Does the vendor sample the data or collect them exhaustively? Is the population sampling methodology sound? In the latter case, who is the original data vendor? Are they reliable?
*   Completeness. Are there data that are obviously missing from the dataset such as, for example, intermittently missing prices? Are there data that are non-obviously missing, such as, for example, unrecorded consumer credit card transactions? Both of these questions must be answered before performing exploratory analysis of the data. If there are issues, these need to be addressed with the vendor.
*   Quality assurance. How does the provider ensure that the data it collects or subjects are consistent and of good quality? Does it have checks for change points in the data (non-stationarities)?
*   Point-in-time versus restated data. Does the provider offer data collected as of a certain date, without changing them at a later date, based on corrections and company updates? This is an instance of data leakage, which we will cover in more detail later.
*   Transformations. Data are often transformed by the vendor. Transformations are: removal of missing data, winsorization and removal of outliers, end-of-period price calculations (last transaction, mid, bid-ask price, weighted average). These transformations should be documented, evaluated, and if possible, verified by the research analyst.

Exploring alternatives and complements. Always ask the following common-sense questions: can we obtain better data, across the following three dimensions. First, are there providers offering larger coverage (i.e., the same data but for more entities or more periods)? Second, are there providers offering more granular data? Third, are there providers with better data? For example, if data are collected from broker-dealers, the alternative provider has an agreement with a larger number of participating broker-dealers. Third, can we obtain complementary data? If data are flows that quantify activity on a social media platform or a given website, for example, we may want to obtain transactional data that help us estimate short-term revenues of a company, in addition to data that give us a good estimate of their costs.

**8.1.2 Research Process**

Every researcher has their own research process. This is part of their competitive advantage; it’s indeed part of what they are, of thoughts and learned lessons accumulated over a lifetime of experiences and of studying. It would be futile to superimpose the author’s own research process to that of the reader. Just as futile would be to recommend that the reader adopt some universal method, part of basic hygiene. Consider these akin to the prompt to never leave home without wearing underwear.
*   Data leakage. The first recommendation is to avoid data leakage. The definition of data leakage is the presence in the training data, the data available up to period $t-1$, of data that are referenced in time $t$, i.e., returns in period $t$, and that are contained in the test data (the data in period $t$ or later). Data that are referenced in time $t$ are used in production on that day. Detecting data leakage is more art than science, and it requires both a deep knowledge of the data (see above) and of the problem at hand. Below are a few examples.
    *   Survivorship bias. This is the most common form of data leakage. It is the practice of conditioning the performance of a strategy on an extended period of time, considering only the stocks that have continuously traded during this period, i.e., the surviving stocks at the end of the backtest, we are subject to survivorship bias. Stocks are most often delisted because of mergers or acquisitions, or because they become illiquid, or they do not meet some additional criteria for being listed on an exchange. Removing them biases the investment sample toward outperformers with different characteristics than those of the broader investment universe. As a point in time, the stocks in the universe, their characteristics, and size are larger than the universe.
    This is the simplest and most impactful instance of data leakage. The remedy to this issue is to: (a) employ a universe that is updated in time; (b) employ a universe that is static in time. By (a) specify a listing and delisting date in the backtest for the event of a delisting. For example, one could assume that the entire investment is written off. Note that the methodology in (a) should be specified before backtesting. (b) means that the universe is chosen based on the result of a backtest is also an instance of data leakage, and it should be avoided. Criteria for inclusion are indeed not straightforward to specify. A common recommendation is to use an investment universe defined by a commercial provider, like Russell 3000 Index, S&P500, MSCI benchmarks, or commercial factor models investment universes. Note that benchmark components are always announced before (“announcement date”) the effective date of the change. The performance of a stock is often affected by the announcement of an inclusion or exclusion. In your backtest, you may want to capture this information, in order to assess how much of the performance of your quantitative strategies is affected by benchmark rebalancing events.
    *   Financial statement information. Financial statement information for a given quarter or year should be included in the backtesting data on the day (or the day after) of their public release, not on the date of the statement data itself.
    *   Point-in-time versus restated data. Data in the backtest on a given date should always be the most recent data available as of that date. If a 10-Q (quarterly financial report) is restated because of a mistake, the backtest should not use the restated data. The decision must be made by allowing the variance of errors in its input data.
    *   Price adjustments. Shares are regularly split (or reverse-split) into multiple shares. The price of the split share is adjusted accordingly. This occurs when the firm appreciates, to the point one share becomes so expensive that it prevents investors from being able to buy it. In order to preserve liquidity, the firm splits its shares. Stock splits and dividends are usually split-adjusted. This introduces a complication. A low stock price at the distant past indicates that the shares have been split several times in the future, likely because of high returns. The price becomes indicative of future good performance. This is data leakage. We must use adjusted prices only for return calculation. For feature generation, use as-of prices.
    *   Missingness. In certain cases identify unstructured data. In particular, missing data are often imputed because they are either available and up-to-date, or because there is some sensitive information and were redacted. In the latter case, missingness may be suffering from look-ahead bias and is information about returns.
    *   Available information. The number of silly mistakes (in hindsight) that experienced, effective researchers make never ceases to amaze. For example, a stock characteristic available in a dataset had high Information Coefficients (IC). Upon further investigation, it was a stock split conversion factor, not the magic bullet after all. In another example, because of an erroneous $t$ versus $t-1$ conversion error, a researcher included the next-day return in a three-month momentum factor definition, also causing a false positive.

Shrinkage and regularization. These techniques are ubiquitous, but, while missing a solid foundation, we hard to argue against.
Have a theory (if any). It is preferable to have a theory for every anomaly and to pre-register the predictions of the theory before the backtest. For example, many papers in the academic literature are based on a theory for the development of the factor. In De Franco and Pedersen (2014) when they analyze the beta anomaly. With a theory as a guide, it is easier to choose a security characteristic among many possible ones, reducing the number of strategies tested. The theory may not be correct and the result would be noise in it, and it is possible to critique and revise the characteristic, which is maybe not desirable (it would be nice if we got it right the first time) but necessary.
Experimental design. Document your assumptions and make sure you can reproduce and revise them at any time.
Use the same setting in backtesting and in production as much as possible. By this we mean that we should use the same point-in-time data, but also the same optimization formulations, the same market impact model, and the same codebase.
Calibrate the market impact model. When we perform a backtest, the market impact model has a “descriptive” role. It is not possible, however, to verify the realized market impact on historical data. In order to run a live backtest, performance has to be simulated out of sample. A market impact model of fair value, especially one provided by a vendor or a partner. Instead, calibrate its parameters against live performance of the current version of your strategy, so that realized PnL is as close to simulated PnL as much as possible.
Include borrow costs. As part of the effort to align production and simulated PnL, one should take borrow costs, for shorted securities, into account, since they can have a material impact on PnL. This is one of the main challenges. Historical borrow rates are not readily available historically. The researcher may have to approximate them, or predict them on the basis of security characteristics. Another complication, albeit minor in impact, is the tax treatment of dividends. When firms are owned by the investor, they are subject to taxation. When the investor is short the security, the treatment of dividends is more complex. In practice, tax-dividend treatment is usually ignored. There is a conceptual difference in backtests, so it is not well understood. Still, be aware.
Define the backtesting protocol beforehand. A backtesting protocol is the sequence of actions and decisions that lead to assessing the performance of a strategy. It is the subject of the next section. For the sake of this list of folk precepts, it is sufficient to say that the backtesting protocol should be charged to the client, for a good reason. And if it is a new design, tested, and evaluated as part of your process goes under the new protocol.
Define the dataset being used beforehand. If dataset selection is seen as part of the backtesting protocol, the heuristic follows from the previous point. The difference is that data and new data become available every day, both in the form of live data, and of extensions to historical dataset. Researchers may be prone to include datasets that confirm their findings, and ignore those that do not. Ignoring new data would be suboptimal, and including them selectively may lead to the wrong conclusions. Use your judgment and research integrity, which no theorems can help.

---

**8.2 The Backtesting Protocol**

**8.2.1 Cross-Validation and Walk Forward**

Evaluating trading strategies bears similarities with statistical model selection (Hastie et al., 2008). We have a family of strategies, a statistics, a loss, or a model, and a performance metric, such as Sharpe Ratio or PnL. We want to select the strategies themselves may depend on several parameters. Two evaluation schemes are most common. The first one is cross-validation (Hastie et al., 2008, Ch. 7; Mohri et al., 2018, Ch. 3). The available data is split into a training dataset and a holdout dataset. The model is trained on the training dataset, and its performance is evaluated on the holdout dataset. The estimate of the performance is then the average of the performance on the holdout dataset. The training dataset is split into $K$ equal-sized samples (“folds”). For the buffer between training and holdout datasets, we may want to separate the folds by a short buffer (for equities, just one or two days) to decrease dependence between the folds. The model is trained on $K-1$ folds, and its performance is evaluated on the remaining fold. This is repeated $K$ times. We perform $K$ estimation-evaluation cycles. The parameters are estimated on each of the possible combinations of $K-1$ folds, and the performance of the model is evaluated on the remaining fold using the optimized parameter; see Figure 8.2. The final estimate of the performance is then the average of the $K$ performances.
A scheme is shown in Figure 8.3. There are several considerations in using cross-validation for financial applications. First, the samples are not independent. The time dependence is reflected in the returns themselves. We know that serial dependence in returns is weak. However, the volatility of returns is serially dependent. If we shuffle the data (or some time series), it is possible to remedy this by keeping the order intact in the folds and the errors are serially uncorrelated (Bergmeir et al., 2018; Cerqueira et al., 2020). This is not the only issue. Financial data are non-stationary. For example, consider the inclusion of a security recently merged in a portfolio. This may create a structural break. If the validation fold precedes temporally the training fold, these past returns are in the validation fold and we are incurring a typical instance of data leakage: the predictors directly contain information about the target. For example, a momentum strategy is a problem. But some strategies, like those usually produced by machine learning, who base their judgment on past returns. Like momentum, we may have leaked target data into the training set. Besides the temporal dependencies, there is another practical obstacle to $K$-fold cross-validation in the influential book, Hastie et al. (2008, Section 7.10). The $K$ estimation-evaluation cycles are computationally very expensive. Cross-validation. Predictive variables (be they alpha or factors, in our framework) should not be screened in advance. This is often not the case in practice: the predictiveness of signals in fully-fledged strategies is tested separately.

(Right Sidebar from Page 351)
Perform cross-validation enough times on different classes of models, and you will inevitably obtain favorable results. The holdout dataset is meant to serve as a check against this overfitting. However, the holdout dataset is not greater than the training set. In the case of financial data, it is inevitable to cycle through several refinements and model revisions, so that the holdout sample performance becomes just another variable to be optimized, instead of a performance check to be run only once.

[Image: Figure 8.2: A scheme of the cross-validation procedure. Dashed boxes are validation folds, while lighter boxes are training folds. Shows a timeline with alternating Training Dataset and Holdout Dataset segments.]

[Image: Figure 8.3: A scheme of the cross-validation procedure. Data is split into two sets: Cross-Validation is performed on the first one (training dataset), to estimate the expected performance of a strategy. The model is then optimized on the entire training dataset, and validated on the second one (validation dataset). Shows a flow: Data -> Cross-Validation -> Training Dataset -> Best Parameters -> Holdout Dataset -> Final Training -> Final Validation.]

As an example we help illustrate the perils of cross-validation. We base $N=1000$ points. We simulate iid asset returns with $r_{it} \sim N(0, \sigma^2)$ with $\sigma = 0.01$. We introduce $P$ random features. These features are also iid random and are design not predictive of returns: $B_{ikt} \sim N(0,1)$.[1] These random features are to design not predictive of returns. The backtest consists of a fixed-fold cross-validation, to estimate the performance of the strategy. We choose $K=5$, and compute the IC and Sharpe Ratio of the strategy. We repeat the process on 1000 simulated datasets. Below are the results for two scenarios:
1.  The first one is the “many periods, few predictors” case: we set $T=5000$ (years of daily data) and $P=2$. Two predictors because one would have felt too lonely.
2.  The second one is the “few periods, many predictors” case: we set $T=250$ (1 year of daily data) and $P=500$, not nearly as many as we meet in practice.

The frequency histograms of the simulations are shown in Figure 8.4. Some summary statistics of the simulations are shown in Table 8.1. Percentages are close to zero, with the exception of the last column. The percentage of samples whose Sharpe Ratio passes the 1% significant level is shown in the last column of the table. Frequency histograms for the two simulated scenarios; the conversion IC to Sharpe Ratio is $\text{SR} = \text{IC} \times \sqrt{252 N}$.

[Image: Figure 8.4: Cross-validated Sharpe for (a) Scenario 1, (b) Scenario 2.]

**TABLE 8.1**
Backtesting results for the two simulated scenarios; the conversion IC to Sharpe Ratio is $\text{SR} = \text{IC} \times \sqrt{252 N}$ (252 trading days).

| T    | P   | "Holdout" p | Mean (SR) | Stdev (SR) | % passing |
| :--- | :-- | :---------- | :-------- | :--------- | :-------- |
| 5000 | 2   | 0.0187      | 0.02      | 0.07       | 1.2       |
| 250  | 500 | 0.04        | 0.04      | 1.4        | 19        |

A remedy to the data leakage issues arising in cross-validation is walk-forward backtesting (Pardo, 2012). In this scheme, we use historical data up to period $t$ and target returns for period $t+1$, see Figure 8.5. The scheme is as close as possible to the production process. It addresses two drawbacks of cross-validation for time-series—serial dependence and risk of data leakage—and it also allows us to naturally incorporate time-varying parameters as the environment changes. These advantages are complementary to cross-validation. As a result, it is often the case that signals, or simplified strategies, are first tested using cross-validation, and then tested “out of sample” in a walk-forward test. This is not ideal, however, since it has an opportunity cost caused by the delay in running the strategy in production. Walk-forward has an additional important drawback: it uses less training data than cross-validation. On the other side, the holdout dataset is large. It is also possible that the parameters of the model have been identified, and only a few new parameters need to be optimized. This drawback becomes negligible. Two additional settings in which walk-forward does not suffer from data limitation when: (a) data are plenty, (b) in the case of high-frequency trading. (b) data are scarce, but the strategy is very simple. Walk-forward is a necessary step in the validation of a strategy, and in its preparation for production.

(Right Sidebar from Page 355)
[Image: Figure 8.5: Two common walk-forward schemes. The top one uses fixed-length training data, thus keeping the estimation procedure comparable. The bottom one uses all of the past data available, possibly weighting data differently based on the interval from the estimation epoch.]

Summing up, neither cross-sectional nor walk-forward schemes are without flaws. Ideally, we would like a protocol with the following features:
1.  non-anticipatory (immune from data leakage);
2.  taking into account serial dependency;
3.  using all data (or possibly multiple testing if $K$ is large or the number of signals);
4.  providing a rigorous decision rule.
Walk-forward meets the first two requirements; cross-validation meets the third. Neither meet the last two. The next section introduces a novel backtesting protocol, the Rademacher Anti-Serum (in short, RAS), which meets these requirements.

---

**8.3 The Rademacher Anti-Serum (RAS)**

**8.3.1 Setup**

We will be concerned with testing the performance of strategies and signals.
1.  Strategies are the walk-forward simulated returns $r_{S,t}$. This is scored by the predicted volatility, which we denote by $\hat{\sigma}_{S,t}$, so that their average equals the empirical Sharpe Ratio for strategy $S$. In this chapter, we will use the notation $\hat{\theta}_S$ for the Sharpe Ratio.
2.  Signals are time-series of weights. For signals, we consider the Information Coefficient (IC) for the signal at time $t$, which is defined as the cosine of the angle (their cosine similarity) between the alpha vector predicted by signal $S$ in period $t-1$, and the idiosyncratic returns in the same period $t$.
The definitions are below:
$$ \text{IC}_{S,t} = \frac{\mathbf{w}_{S,t-1}^T \mathbf{r}_t}{\|\mathbf{w}_{S,t-1}\| \|\mathbf{r}_t\|} \quad \text{(Sharpe Ratio)} $$
$$ \text{IC}_{S,t} = \frac{\mathbf{w}_{S,t-1}^T \mathbf{r}_t}{\|\mathbf{w}_{S,t-1}\| \|\mathbf{r}_t\| \hat{\sigma}_{S,t} \hat{\sigma}_{R,t}} \quad \text{(Information Coefficient)} $$
We also denote by $|\mathcal{S}|$ the number of strategies. The interpretation will be clear from the context. In one case, the dataset needed for the analysis is a $T \times N$ matrix $\mathbf{X}$. Rows denote observations of a certain timestamp and columns denote strategies, whose set we denote $\mathcal{S}$. For notational simplicity, the $s$-th strategy is denoted by $X_s$, its $t$-th column is $X_{s,t}$. The following results are important as they are quite general. We have two justifications for the assumptions. The first one is empirical. Serial dependence is small for returns observed at daily frequencies or lower.[2] The second one is that our framework can be extended to the case of time-dependent observations. In this case, we assume that the time series of returns is $\alpha$-mixing, i.e., the autocorrelation up to lag $k$, then replace the original time series with $N/k$ non-overlapping, contiguous averages of blocks:
$$ (X_{s,1}, \dots, X_{s,k}), \dots, (X_{s, (N/k-1)k+1}, \dots, X_{s,N}) $$
We employ the following notation. We let the joint distribution of $X_s$ be $P$. Let $D = \otimes_{s \in \mathcal{S}} P_s$. The joint probability distribution on the space of $T \times N$ matrices in which the element $X_{s,t} \sim P_s$ has independent, identically distributed (iid) rows, each drawn from $P$.

(Right Sidebar from Page 356)
The expected value of $X_s$ is denoted by $\theta_S = E_P[X_s]$. This is the true strategy/signal performance. Define $\hat{\theta}_S = \frac{1}{T} \sum_{t=1}^T X_{s,t}$ as the vector of column averages of $\mathbf{X}$.
(8.1)
$$ \hat{\theta}_S = \frac{1}{T} \sum_{t=1}^T X_{s,t} $$
which is the expected value of $X_s$ according to the bootstrap distribution.
Let $\mathbf{\epsilon}$ be a Rademacher random vector in $\mathbb{R}^T$, $T$-dimensional random vector whose elements are iid, and take values $+1$ or $-1$ with probability $1/2$. The Rademacher complexity of $\mathcal{S}$ (Mohri et al., 2018)
(8.2)
$$ \hat{\mathcal{R}}_T(\mathcal{S}) = E_\epsilon \left[ \sup_{s \in \mathcal{S}} \frac{\mathbf{\epsilon}^T \mathbf{X}_s}{T} \right] $$
Before stating a rigorous result linking this quantity to a bound on performance, we focus our attention on its interpretation. Specifically, we can interpret $\hat{\mathcal{R}}_T$ as follows.
*   As the resemblance to random noise. Consider $S$ as a random covariate. We can interpret $\hat{\mathcal{R}}_T$ as the expected value of the highest covariance of the performance measure of a strategy to random noise, $E_\epsilon$. On average, for every set of $+/-1$ indicators, there is at least a strategy that covaries with it, then “we can do no wrong”: for every realization of random noise, there is a strategy that performs well. This implies that, if $E_\epsilon[\mathbf{\epsilon}^T \mathbf{X}_s / T] \approx 1$, if we interpret the $X_s$ as predictions for epoch $t$, then this means that for every sequence of events $\epsilon_t$, we have a strategy that predicts them well.
*   As generalized two-way cross-validation.[3] For sufficiently large $T$, the sets of positive and negative indicators are of size roughly $T/2$. Let $S^+$ be the set of $T/2$ periods where $\epsilon_t = +1$, and $S^-$ the other periods. Rewrite the term inside the sup as
    $$ \frac{1}{T} \sum_{t=1}^T \epsilon_t X_{s,t} = \frac{1}{2T} \sum_{t \in S^+} X_{s,t} - \frac{1}{2T} \sum_{t \in S^-} X_{s,t} = \frac{1}{2} (\hat{\theta}_{s,S^+} - \hat{\theta}_{s,S^-}) $$
    $$ \hat{\theta}_{s,S^+} := \frac{1}{T/2} \sum_{t \in S^+} X_{s,t} $$
    $$ \hat{\theta}_{s,S^-} := \frac{1}{T/2} \sum_{t \in S^-} X_{s,t} $$
    For strategy $s$, this is the discrepancy in average performance measured on two equal-sized random subsets of the observations. By taking the sup across strategies, we are selecting the worst case. We estimate performance on a subset, and get a very different result on the remaining subset. And if the discrepancy is high for each random subset, at least one strategy that performs well. This means that there is always a strategy that performs well. The Rademacher complexity is a measure of how well a strategy performs “out of sample” when the performance is measured on a random subset of the data, and then averaged over all possible random subsets.
*   As measure of span of possible performances. We interpret $\mathbf{X}_s \in \mathbb{R}^T$ as a “random direction” chosen at random in $\mathbb{R}^T$. The vector has Euclidean norm equal to $\sqrt{T}$. In the case where the performance measure is the IC, the vector $\mathbf{X}_s / \sqrt{T}$ is on the unit sphere, and is strongly concentrated around this value. The empirical Rademacher complexity $\hat{\mathcal{R}}_T(\mathcal{S})$ is then approximately equal to
    $$ E_\epsilon \left[ \sup_{s \in \mathcal{S}} \frac{\mathbf{\epsilon}^T \mathbf{X}_s}{\|\mathbf{X}_s\|} \right] $$
    This can be interpreted in the following way. We have a set of $N$ unit vectors $\mathbf{X}_s / \|\mathbf{X}_s\|$. We pick a random direction in the ambient space and observe the maximum collinearity (expressed as the cosine similarity) of this random direction to the vectors. The expected value of this maximum collinearity is the Rademacher complexity. If the vectors are very well spread, it is likely that one of them is very close to the random direction. If the vectors are all copies of the same vector, the maximum collinearity is not very well. If, conversely, these vectors are all orthogonal, we have $N$ unit vectors. The Rademacher complexity is a geometric measure of how much the vectors $\mathbf{X}_s$ “span” $\mathbb{R}^T$. [4]
One interesting characteristic of the Rademacher complexity is that it takes into account dependencies among strategies. If, for example, we had $N$ strategies to test, but $N-1$ are copies of the first one, then we would not need the Rademacher complexity. However, if the $N$ strategies are uncorrelated from each other, then the Rademacher complexity is high, indicating higher likelihood of overfitting.

**8.3.2 Main Result and Interpretation**

The thrust of this section is to provide a uniform, additive “haircut”. This term we subtract from the empirical performance to the performance of strategies. In other words, for each strategy $s$ we have an empirical performance $\hat{\theta}_s$, given by Equation (8.1). In the case of excess returns, this is the empirical Sharpe Ratio. Then, we can establish a probabilistic guarantee on the true Sharpe Ratio of strategy $s$. With probability, say, greater than $1-\delta$, the Sharpe Ratio of the strategy is greater than $\hat{\theta}_S - \text{haircut}$.
(8.3)
$$ \theta_S \ge \hat{\theta}_S - 2\hat{\mathcal{R}}_T(\mathcal{S}) - \sqrt{\frac{2\log(2/\delta)}{T}} $$
(data snooping) (estimation error)
The result is described in Procedure 8.1.

**Procedure 8.1: Rademacher Anti-Serum for signals**
1.  Backtest all the strategies using a walk-forward protocol. Let $\mathbf{X}_s \in \mathbb{R}^T$ be the time series of Information Coefficients of strategy $s$ at time $t$.
2.  Compute $\hat{\mathcal{R}}_T(\mathcal{S})$, as defined in Equation (8.2).
3.  Compute $\hat{\theta}_S(\mathbf{X}_s)$
    for all $s \in \mathcal{S}$, $t=1, \dots, N$
4.  $$ \hat{\theta}_S \ge \hat{\theta}_S - 2\hat{\mathcal{R}}_T(\mathcal{S}) - \sqrt{\frac{2\log(2/\delta)}{T}} $$
    with probability greater than $1-\delta$.
Now, we consider the case for Sharpe analysis. The formula is similar, but with a different, reset estimation error.
(8.4)
$$ \hat{\theta}_S \ge \hat{\theta}_S - 2\hat{\mathcal{R}}_T(\mathcal{S}) - \sqrt{\frac{2\log(2/\delta) (1+\hat{\theta}_S^2/2)}{T}} $$
(data snooping) (estimation error)
The proofs are in the Appendix, Section 8.5.

**Procedure 8.2: Rademacher Anti-Serum for Strategies**
1.  Backtest all the strategies using a walk-forward protocol. Let $\mathbf{X}_s \in \mathbb{R}^T$ be the matrix with Information Ratio of strategy $s$ at time $t$.
2.  Compute $\hat{\mathcal{R}}_T(\mathcal{S})$, as defined in Equation (8.2).
3.  Compute $\hat{\theta}_S(\mathbf{X}_s)$
    for all $s \in \mathcal{S}$, $t=1, \dots, N$
4.  $$ \theta_S \ge \hat{\theta}_S - 2\hat{\mathcal{R}}_T(\mathcal{S}) - \sqrt{\frac{2\log(2/\delta)}{T}} $$
    with probability greater than $1-\delta$.

We focus on the interpretation of the claims. The theorem states that the lower bounds on IC and Sharpe hold simultaneously, at least with probability $1-\delta$. Moreover the statement holds for any finite $T$; no asymptotic approximation is invoked. The true expected performance differs from the empirical performance because of two non-negative terms:
*   The first is the term $2\hat{\mathcal{R}}_T$. This is the data-snooping term. The larger the number of strategies, the higher the $\hat{\mathcal{R}}_T$, because $\sup$ is strictly increasing in the number of strategies. Moreover, as we saw in Section 8.3.1, higher dependency among strategies implies lower $\hat{\mathcal{R}}_T$. This is the first case where we test multiple replicas of the same strategy, $\hat{\mathcal{R}}_T$ is zero. To provide some intuition about the behavior of Rademacher complexity, we consider a set of strategies with normally distributed returns with mean $\mu_S$ and variance $\sigma_S^2$. We vary the number of strategies, and the correlation, and increases in the number of strategies. Given the data matrix $\mathbf{X}$, the quantity $\hat{\mathcal{R}}_T$ is estimated via simulation. An upper bound for this quantity is given by Massart’s lemma:
    $$ \hat{\mathcal{R}}_T \le \sqrt{\frac{2 \log N}{T}} \max_s \|\mathbf{X}_s\| $$

[Image: Figure 8.6: Rademacher complexity for poor strategies, with iid Gaussian returns and variable pairwise correlation. Estimate based on $10^4$ samples.]

*   The second is the estimation term. For some situation, consider the case of $T$ iid normal random variables $\theta_S$ with mean $\theta$ and unit variance. Their average $\hat{\theta}_S$ is distributed as a normal distribution with standard deviation $1/\sqrt{T}$. What if $\theta_S$ is quantile of the distribution? For example, the $5\%$ quantile of the distribution. For a normal distribution with zero mean and standard deviation $1/\sqrt{T}$, $F^{-1}(\delta) = \sqrt{2 \log(1/\delta)} / \sqrt{T}$, and Cumulative Distribution Function $F$.
    (8.5)
    $$ F^{-1}(\delta) = \frac{\sqrt{2\log(1/(2\sqrt{2\pi\delta}))}}{T} $$
    This is similar to, up to constants, to the estimation errors in Equation (8.3) and (8.4). In the limit $T \to \infty$, the estimation error in both procedures approaches 0.
    The estimation error above is independent of $N$ for bounded distributions and is $\propto 1/\sqrt{N}$ for sub-Gaussian ones. An argument for this property is based on the following. Consider the following special case. Given $N$ iid Gaussian random variables $X_i \sim N(0,1)$. The empirical Sharpe Ratios of the $N$ strategies are also iid, $\sim N(0, 1/T)$. It can be shown (Van Handel, 2016; Kamath, 2020) that:
    of $s$. This seems a loose bound, compared to the standard formula for the standard error of the Sharpe Ratio (i.e., $\text{stdev} = \sqrt{(1+SR^2/2)/T}$). For a strategy with Sharpe Ratio equal to 2, the estimation error haircut is
    $$ F(\delta) \sqrt{(1+SR^2/2)/T} \sqrt{251} \approx 1.7 $$
    The constant of the data-snooping term is also conservative, since in the proof we rely on a chain of inequalities to obtain a bound.
    I will close with the wise words of a former colleague. The path connecting theory to practice is often tortuous. But when it is straight, it is usually short. I hope that theory is insightful, applicable, useful, and not too complicated. The bound of Procedure 8.2 will take the form:
    $$ \theta_S - \hat{\theta}_S \ge -2\hat{\mathcal{R}}_T - b \sqrt{\frac{2\log(2/\delta)}{T}} $$
    with positive parameters $a,b$.

When the SR is zero, the maximum (and high quantiles) of the Sharpe Ratios grows as $\sqrt{2 \log N / T}$. This is not to say that the estimation error on the individual SR should be equal to $\sqrt{2 \log N / T}$. For some positive $\alpha$, in addition, we should have a term that captures the tail behavior of error, $T^{-\alpha}$. For large $N$, the dominant term of the estimation error is $\sqrt{2 \log N / T}$.
$$ \sqrt{2 \log N / T} + 2 \log(2/\delta)/T $$
which is majorized by
$$ \sqrt{2 (\log N / T + \sqrt{2 \log(2/\delta)/T})} $$
The first term is the $\hat{\mathcal{R}}_T$ growth term, and the second one is the term accounting for the confidence interval that we saw in Equation (8.4).
The procedure is operationally simple: simulate all possible strategies in a walk-forward manner. There should be no look-ahead bias: the strategies should be formulated without looking at future data. The strategies should be based on good modeling practice. As we mentioned before, “best practice” means that all strategies should be documented and should run in parallel to the production strategy. Then, estimate the Rademacher complexity of matrix $\mathbf{X}$ by the expectation in the definition of that statistic. The Rademacher complexity is easy to compute for small $N$. For large $N$, it can be computed for even larger sets of strategies using tools from numerical analysis.
The RAS procedure for signals uses the worst case $|\theta_S| \le 1$. In practice, however, it is extremely unlikely to observe IC close to one. IC greater than 0.1 is extremely unlikely. If we assume $|\theta_S| \le C \le 1$, and apply Theorem 8.3, the estimation term becomes smaller, by a factor of $c$.
$$ \text{“estimation error”} = 2 \epsilon_S \sqrt{\frac{\log(2/\delta)}{T}} $$
Consider some realistic parameters: $\epsilon_S = 0.02, \delta = 0.01$, and $T=2500$. Then the estimation error is about 0.002.
In the RAS procedure for strategies, the formula for the estimation error is rather simple and the constant factors are probably not too conservative. The error depends on the square root of $T$. For example, with $\delta=0.01, T=2500$ and $N=1.0\%$, then the estimation error is 0.03, corresponding to an annualized estimation error of $5\%$. The estimation error for $\delta=0.05$.
3.  The percentage of positive strategies, as per Equation (8.3).
4.  The percentage of Rademacher positives.
5.  The percentage of true positive strategies. This percentage is either 0 or 20%.
Figures 8.7 and 8.8 show the results for normally and subdistributed returns, respectively. We interpret the results below:
1.  For a fixed distribution of the population Sharpe, the maximum empirical Sharpe Ratio $\max_s \hat{\theta}_S$ is predictably increasing in $N$. It is also increasing in $\beta$ because the “effective” number of assets decreases, as the strategies are more correlated, and it is decreasing in $T$, by the Central Limit Theorem.
2.  Everything else equal, the Rademacher complexity is decreasing in $T$, by the Central Limit Theorem. It is interesting to compare the true $\hat{\mathcal{R}}_T$ from the simulations with the bound from Massart’s lemma. In Table 8.2, we report the highest $\hat{\mathcal{R}}_T$ from Table 8.3, and Massart’s bound is at most 15% higher than the observed Rademacher complexity.

**TABLE 8.2**
Comparison of “sup hat” and Massart’s bound

| N    | T    | "hold-italic cap r hat" | "hold-italic cap r hat" Massart's Bound |
| :--- | :--- | :---------------------- | :-------------------------------------- |
| 500  | 2500 | 0.020                   | 0.023                                   |
| 500  | 5000 | 0.017                   | 0.019                                   |
| 5000 | 2500 | 0.027                   | 0.030                                   |
| 5000 | 5000 | 0.021                   | 0.023                                   |
| 5000 | 7500 | 0.015                   | 0.018                                   |

---

**8.4 Some Empirical Results**

**8.4.1 Simulations**

Let us see how this approach performs in a simulated setting first. We first consider strategies whose returns are normally distributed, are iid (either across strategies or across time periods), and with correlation $\rho$. Specifically, the return of strategy $s$ in period $t$ is given by
$$ r_{s,t} = \rho f_t + \sqrt{1-\rho^2} \epsilon_{s,t} $$
with $f_t \sim N(0,1)$, $\epsilon_{s,t} \sim N(\mu_S, 1)$. $\mu_S$ are constants for different values of the population Sharpe Ratios, for different numbers of strategies, and for expected returns (and therefore non-annualized Sharpe Ratios) equal to 0 and 0.1. For each simulation, we report the maximum empirical Sharpe Ratio $\max_s \hat{\theta}_S$, the Rademacher complexity, the estimation error, and the percentage of detected positive strategies, i.e., of strategies whose lower bound of the 95% confidence interval of the true Sharpe Ratio. We also report the percentage of “Rademacher positive strategies,” i.e., the strategies that exceed the data-snooping haircut alone. In formula,
(8.6)
$$ \hat{\theta}_S - 2\hat{\mathcal{R}}_T(\mathcal{S}) > 0 \quad \text{(“positive”)} $$
(8.7)
$$ \hat{\theta}_S - 2\hat{\mathcal{R}}_T(\mathcal{S}) > 0 \quad \text{(“Rademacher positive”)} $$
We perform simulations with returns distributed both according to a Gaussian distribution and to a t-distribution with five degrees of freedom. The latter aims to approximate heavy-tailed returns. The simulations are performed for all possible combinations of the following parameters:
1.  Correlation: $\rho \in \{0.2, 0.8\}$.
2.  Number of strategies: $N \in \{500, 5000\}$.
3.  Number of periods: $T \in \{2500, 5000\}$.
4.  Population Sharpe: We consider two cases. In the first one, all strategies have SR = 0. In the second one, 80% have SR = 0, and 20% have SR = 0.2 (the Sharpe Ratio is not annualized).
For each effect, we use eight simulations. For each simulation we compute a historical return matrix $\mathbf{X}_s$. For each simulation we compute the following outputs:
*   $\max_s \hat{\theta}_S$, the maximum realized Sharpe Ratio across the strategies.
*   $\hat{\mathcal{R}}_T$, the Rademacher complexity of $\mathbf{X}_s$.

**TABLE 8.3**
Simulations for normally distributed returns

| rho | N    | T    | "hold-italic max over hat theta" | "hold-italic cap r hat" | "hold-italic est err" | Error % "hold-italic pos rad" | Error % "hold-italic pos true" |
| :-- | :--- | :--- | :------------------------------- | :---------------------- | :-------------------- | :---------------------------- | :----------------------------- |
| 0.2 | 500  | 2500 | 0.2                              | 0.019                   | 0.039                 | 0.0                           | 0.0                            |
| ... | ...  | ...  | ...                              | ...                     | ...                   | ...                           | ...                            |
| 0.8 | 5000 | 5000 | 0.2                              | 0.230                   | 0.010                 | 20.0                          | 20.0                           |

**TABLE 8.4**
Simulations for t-distributed returns

| rho | N    | T    | "hold-italic max over hat theta" | "hold-italic cap r hat" | "hold-italic est err" | Error % "hold-italic pos rad" | Error % "hold-italic pos true" |
| :-- | :--- | :--- | :------------------------------- | :---------------------- | :-------------------- | :---------------------------- | :----------------------------- |
| 0.2 | 500  | 2500 | 0.2                              | 0.016                   | 0.039                 | 0.0                           | 0.0                            |
| ... | ...  | ...  | ...                              | ...                     | ...                   | ...                           | ...                            |
| 0.8 | 5000 | 5000 | 0.2                              | 0.231                   | 0.010                 | 20.0                          | 20.0                           |

3.  The data-snooping term and the estimation error term have the same magnitude.
4.  In the null Sharpe case (all strategies have zero Sharpe Ratio), the percentage of detected positive cases (1% pos) is zero or nearly zero in all cases: there are no false positives.
5.  In the alternative Sharpe case (20% of strategies have Sharpe Ratio equal to 0.2), the percentage of true positives is smaller than the percentage of true positives. All detected positives are in fact true positives: the False Discovery Rate (FDR), defined as the ratio of false positives to all detected positives, is zero. In a few cases the percentage of positives is somewhat lower than the percentage of true positives.
6.  The percentage of Rademacher positives is much closer to the percentage of true positives. The FDR is still zero in this case.

---

**8.4.2 Historical Anomalies**

It is of independent interest to analyze anomalies published in the academic literature. We consider two sources of returns: the first (Jensen et al. (2021)) is a dataset of factor anomalies introduced by Hou et al. (2015), and test their replicability and out-of-sample performance. The dataset contains published characteristics, and it is not possible to know which characteristics were tested prior to publication. In principle, their variants should be included in the data. Had they been included, the Rademacher complexity of the dataset would have been higher. As criteria for inclusion, we required that the factor have at least 10 years of trading history, and that they be produced on the last day in the dataset, December 31, 2021. We perform the analysis at the country level. Table 8.5 presents the results. The table shows the number of strategies, the average percentage of true positives, which is common across a priori. The United Kingdom and Hong Kong are the only markets where a positive percentage of factors meet the Rademacher bound, Equation (8.6).

**TABLE 8.5**
Summary data for the factors in Jensen et al.’s database

| Country | N   | "hold-italic T" | max over hat theta | "hold-italic cap r hat" | Error % "hold-italic est err" | % "hold-italic pos rad" | % "hold-italic pos true" |
| :------ | :-- | :-------------- | :----------------- | :---------------------- | :---------------------------- | :---------------------- | :----------------------- |
| AUS     | 153 | 3584            | 0.058              | 0.046                   | 0.030                         | 0.0                     | 0.0                      |
| ...     | ... | ...             | ...                | ...                     | ...                           | ...                     | ...                      |
| ZAF     | 151 | 2876            | 0.092              | 0.047                   | 0.035                         | 0.0                     | 0.0                      |

Another source of factor return data is curated by Andrew (Chen and Zimmermann, 2022). Among the anomalies, we select those that were available as of the end of 2021 and had at least 5 years of history. The results are displayed in Table 8.6. The percentage of anomalies that meet the Rademacher bound is about 16%. The smaller percentage is attributable to the fact that the number of periods (days) with complete observations is 3931 (compared to 13,125 for the Jensen, Kelly, and Pedersen dataset) and to the Rademacher complexity being 0.033 (compared to 0.021 for the Jensen, Kelly, and Pedersen dataset).

**TABLE 8.6**
Summary data for the factors in Chen and Zimmermann’s database

| N   | "hold-italic T" | max over hat theta | "hold-italic cap r hat" | Error % "hold-italic est err" | % "hold-italic pos rad" | % "hold-italic pos true" |
| :-- | :-------------- | :----------------- | :---------------------- | :---------------------------- | :---------------------- | :----------------------- |
| 162 | 3931            | 0.126              | 0.033                   | 0.031                         | 0.0                     | 12.3                     |

---

**8.5 Appendix**

**8.5.1 Proofs for RAS**

We use some essential inequalities in the proofs. Standard references are Boucheron et al. (2013) and Vershynin (2018).

**Theorem 8.1 (McDiarmid's Inequality):**
Let $X_1, \dots, X_n$ be independent random variables, and $f: \mathbb{R}^n \to \mathbb{R}$, such that for each $i$,
$$ \sup_{x_1, \dots, x_n, x_i'} |f(x_1, \dots, x_i, \dots, x_n) - f(x_1, \dots, x_i', \dots, x_n)| \le c_i $$
Then, for all $\epsilon > 0$,
$$ P(|f - Ef| > \epsilon) \le 2 \exp \left( - \frac{2\epsilon^2}{\sum c_i^2} \right) $$
Specifically, if $c_i = c$, and with probability greater than $1-\delta/2$,
$$ f < Ef + \sqrt{\frac{nc^2 \log(\delta/2)}{2}} $$
A mean zero sub-Gaussian random variable $X$ is one for which a positive constant $c$ exists, such that the inequality
$$ P(|X| > \epsilon) \le 2 \exp(-\epsilon^2/(2c^2)) $$
holds for all positive $\epsilon$. The parameter $\sigma^2$ is the proxy variance.

**Theorem 8.2 (Generalized Hoeffding's Inequality):**
Let $X_1, \dots, X_n$ be iid random variables with finite sub-Gaussian norms and proxy $\sigma_i$. Then, for all $a_i > 0$,
(8.7)
$$ P \left( \left| \sum_{i=1}^n a_i (X_i - EX_i) \right| > t \right) \le 2 \exp \left( - \frac{t^2}{2 \sum a_i^2 \sigma_i^2} \right) $$
(8.8)
$$ \sup_{s \in \mathcal{S}} (\hat{\theta}_s - \theta_s) \le \phi(\dots, \mathbf{X}_s, \dots) + \frac{2\hat{\mathcal{R}}_T(\mathcal{S})}{T} $$
We apply McDiarmid’s inequality to $\phi$ to obtain the result. In order to obtain a lower bound on $E\phi$, we need an upper bound on $E\hat{\mathcal{R}}_T$. In the inequalities below, we introduce a probability measure $P'$ identical to, and independent from, $P$.
$$ E_P \phi_{sup} = E_P[\hat{\theta}_s] $$
$$ = E_P[\sup_{s \in \mathcal{S}} (\hat{\theta}_s(X) - \theta_s(X'))] $$
$$ = E_P[\sup_{s \in \mathcal{S}} (\hat{\theta}_s(X) - \hat{\theta}_s(X'))] \quad \text{(conditioning)} $$
$$ \le E_P E_{P'} [\sup_{s \in \mathcal{S}} (\hat{\theta}_s(X) - \hat{\theta}_s(X'))] \quad \text{(Jensen)} $$
$$ \le E_P E_{P'} [\sup_{s \in \mathcal{S}} \sum_{t=1}^T \epsilon_t (X_{s,t}(X) - X_{s,t}(X'))] $$
$$ = E_P E_{P'} [\sup_{s \in \mathcal{S}} \sum_{t=1}^T \epsilon_t X_{s,t}(X)] + E_P E_{P'} [\sup_{s \in \mathcal{S}} \sum_{t=1}^T \epsilon_t (-X_{s,t}(X'))] $$
We introduce an additional source of noise (a Rademacher matrix) and have a constant $K$. We can gain in tractability. We can change the signs of each summand by multiplying by some arbitrary factor $K \in [-1,1]$, since the terms are exchangeable.
$$ = E_P E_{P'} [\sup_{s \in \mathcal{S}} \sum_{t=1}^T \epsilon_t X_{s,t}(X)] + E_P E_{P'} [\sup_{s \in \mathcal{S}} \sum_{t=1}^T \epsilon_t' X_{s,t}(X')] $$
$$ \le E_P E_{P'} [\sup_{s \in \mathcal{S}} \sum_{t=1}^T \epsilon_t X_{s,t}(X)] + E_P E_{P'} [\sup_{s \in \mathcal{S}} \sum_{t=1}^T \epsilon_t' X_{s,t}(X')] $$
$$ = 2 E_P \hat{\mathcal{R}}_T $$
since we defined $\hat{\mathcal{R}}$ as the expected value of the Rademacher complexity over the distribution of performance realizations.
We now use McDiarmid again for all $P_s \in \mathcal{S}$.

(Right Sidebar from Page 375)
**Theorem 8.3 (Bounds for bounded performance metrics):**
Assume that $|\theta_{s,t}| \le c_s$ for all $s \in \mathcal{S}$, $t=1, \dots, T$.
For all $n=1, \dots, N$,
(8.8)
$$ \theta_n \ge \hat{\theta}_n - 2\hat{\mathcal{R}} - 3c_n \sqrt{\frac{2\log(2/\delta)}{T}} $$
Proof
The straightforward inequality holds for all $n=1, \dots, N$:
$$ \theta_n - \hat{\theta}_n \ge \sup_n |\theta_n - \hat{\theta}_n| - |\hat{\theta}_n - \theta_n| $$
Define
$$ \Phi := \sup_n |\theta_n - \hat{\theta}_n| $$
$$ \phi := \sup_n (\hat{\theta}_n - \theta_n) $$
We claim that with probability greater than $1-\delta/2$,
(8.10)
$$ \Phi \le E\Phi + c_n \sqrt{\frac{2\log(2/\delta)}{T}} $$
This allows one to deal with $E\Phi \approx E\phi_n - \theta_n$, which is easier. To prove the inequality, note that, for all $X_{s,t} \in [-c_s, c_s]$, $t=1, \dots, T$, $i=1, \dots, N$
(8.11)
$$ |\hat{\theta}_{s,t} (\dots, X_{s,t}, \dots) - \hat{\theta}_{s,t} (\dots, X'_{s,t}, \dots)| \le \frac{2c_s}{T} $$
from which it follows that
$$ |\hat{\mathcal{R}}(\dots, X_{s,t}, \dots) - \hat{\mathcal{R}}(\dots, X'_{s,t}, \dots)| \le \frac{2c_s}{T} $$
Hence, with probability greater than $1-\delta/2$
(8.12)
$$ \hat{\mathcal{R}} \le E\hat{\mathcal{R}} + c_n \sqrt{\frac{2\log(2/\delta)}{T}} $$
Now we employ the union bound on inequalities (8.10) and (8.12) to obtain the claim.

**Theorem 8.4 (Bounds for Sub-Gaussian performance metrics):**
Assume that
$$ P(|X_{s,t}| > \epsilon) \le 2e^{-\epsilon^2/(2\sigma_s^2)} $$
for all $s > 0$, for $n=1, \dots, N$, $t=1, \dots, T$.
Then, for all $s \in \mathcal{S}$,
$$ \theta_s - \hat{\theta}_s \ge -2\hat{\mathcal{R}} - 3\sigma_s \sqrt{\frac{2\log(2N/\delta)}{T}} $$
Proof
Let $a > 0$. We split $\theta_s - \hat{\theta}_s$ into the sum of two terms:
$$ \theta_s - \hat{\theta}_s = g(X^s, a) + h(X^s, a) $$
where
$$ g(X^s, a) := E_P[\sup_{s \in \mathcal{S}} (\hat{\theta}_s(X) - \theta_s(X)) \mathbb{I}(|\hat{\theta}_s(X) - \theta_s(X)| \le a)] $$
$$ h(X^s, a) := E_P[\sup_{s \in \mathcal{S}} (\hat{\theta}_s(X) - \theta_s(X)) \mathbb{I}(|\hat{\theta}_s(X) - \theta_s(X)| > a)] $$
We bound $P(\sup_s |h(X^s,a)| \ge \epsilon)$
By symmetrization
The random variable $h(X_s, a)$ is sub-Gaussian, since it is dominated by $|X_s|$. Both probability $X_s$ and $h$ has the same proxy variance as $|X_s|$. By the general Hoeffding inequality,
$$ P \left( \left| \sum_{t=1}^T h(X_{s,t}, a) \right| > t \right) \le \exp(-T t^2/(2\sigma_s^2)) $$
$$ P \left( \sup_{s \in \mathcal{S}} \left| \sum_{t=1}^T h(X_{s,t}, a) \right| > t \right) \le N \exp(-T t^2/(2\sigma_s^2)) $$
By the union bound,
(8.13)
$$ \hat{\theta}_S \ge \hat{\theta}_S - 2\hat{\mathcal{R}} - 3\sigma_S \sqrt{\frac{2\log(2N/\delta)}{T}} - \frac{a}{T} $$
probability $1-\delta$.
The reduced performance metric is guaranteed to be greater than or equal to the true performance metric with probability $1-\delta$.
The haircut is the sum of a “data-snooping term,” a function of the set of strategies, and an “estimation term,” function of the sampling interval.

**The Takeaways**

*   Strategy performance of systematic strategies is usually validated against historical data. Historical data is often scarce, whereas the number of strategies being tested can be extremely large. We reuse the same data to test and tune strategies.
*   Two core principles of an effective backtesting process are:
    1.  Perform careful data sourcing.
    2.  Use a backtesting protocol that ensures that the historical simulation reproduces the run of the same strategy in the real world.
*   Common backtesting procedures are cross-validation and walk-forward.
*   We propose an alternative procedure: the Rademacher Anti-Serum (RAS).
*   RAS provides a “haircut”, i.e., term that is subtracted from the empirical performance metric of every strategy. This haircut depends on $T$.

**Notes**

1.  [1] Assuming 6.5 trading hours, 252 trading days, and 3000 stocks.
2.  [2] Although, you may argue, this whole book is an exposition of my investment philosophy. Point taken, to an extent. I am providing some building blocks, and you are reshaping and assembling them into something sensible.
3.  [3] The Information Coefficient is formally defined in Section 8.3.
4.  [4] In order to short a security, the investor (or an agent on their behalf, like a broker-dealer) must borrow it first from a lender, who charges interest on the loan.
5.  [5] This is the percentage of simulation samples for which the condition $\text{SR} > 2.3 \sqrt{(1+SR^2)/T}$
6.  [6] The $\hat{\mathcal{R}}_T$ is featured prominently in Section 8.3.
7.  [7] For definitions and use of $\alpha$-mixing, see e.g., Cont (2001) and Taylor (2008).
8.  [8] See Section 7.2 and references therein, for example, Cont (2001) and Taylor (2008).
9.  [9] The estimation error of Equation (8.4) for $\delta=0.05$ and $SR=1.0\%$ is. The estimation error is independent of $N$ is smaller than for $\delta=0.01$.
    $$ 2 \sqrt{\frac{\log(2N/\delta)}{T}} \le 0.6 \times 3.3 \sqrt{\frac{\log(2/\delta)}{T}} $$
    , and is very weakly dependent on $N$.
10. [10] Data downloaded from https://www.aeaweb.org/journals/aer/data/106-12_data.zip, on August 20, 2022.
11. [11] Data downloaded from https://www.aeaweb.org/articles?id=10.1257/aer.20191645, on August 20, 2022.

```

Okay, here is the Markdown compilation for Chapter 9.
     Everand

