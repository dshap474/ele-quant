**Chapter 7**
**Statistical Factor Models**

**The Questions**

1.  How do we estimate factor models when both factor returns and exposures are unknown?
2.  How do we employ Principal Component Analysis (PCA)? What specific adaptations do we need to employ?
3.  How do we interpret statistical models, especially when it comes to the loadings that are less interpretable than those of alternative estimation methods?
4.  How do we reduce factor model turnover?

In the statistical model framework, we assume that we know neither the factor returns nor the exposures; we estimate both. The estimation relies on Principal Component Analysis (PCA). Starting with Chamberlain (1983), this approach has been receiving an increasing amount of attention in the academic literature. Factors, in this context, are often called statistical risk factors or unobserved common factors. When the number of assets is large, there is a clear separation between the few largest eigenvalues and the remaining eigenvalues. The PCA solution constitutes then a good approximation and, in the limit, converges to the true model. In applications, one may question the merit of an approach that, unlike the fundamental and the macroeconomic environment. Developing a statistical model is useful for several reasons:

*   Complementarity. Using several models helps understand the shortcomings of each individual model. We can project an existing model onto a statistical model, or augment it with statistical factors.
*   Optimality. In a portfolio optimization problem, it may have beneficial to use combinations in which we have estimated the total factor variance using different models; or, we could include both variances as constraints.
*   Data. In certain asset classes, firm characteristics or relevant macroeconomic data may not be available. When only returns are available, statistical models are the only option.
*   Availability at Short Time Scales. At certain time scales, such as 1- or 5-minute intervals, fundamental factors may not be relevant.
*   Performance. Statistical models may just outperform the alternatives.

The main disadvantage of statistical models is that the factors are less interpretable than in the alternative estimation methods. The first factor is usually easy to interpret as the market. The second and third ones can find an interpretation. For example, Litterman and Scheinkman (1991) interpret the first three statistical factors as level, steepness, and curvature of the bond yield curve. The situation is less clear for subsequent factors. Another approach is to rotate the statistical factors. In the words of Johnson and Wichern (2007), “Analyses of principal components are more of a means to an end rather than an end in themselves because they frequently serve as intermediate steps in much larger investigations.” This is perhaps true of all factor models, but is certainly truer with regards to statistical models, because of the possible challenges in interpretation.

This chapter starts with a minimal description of the approach. Then, we take a detour in the real world.

---

**7.1 Statistical Models: The Basics**

**7.1.1 Best Low-Rank Approximation and PCA**

Let $\mathbf{R} \in \mathbb{R}^{N \times T}$ be the matrix of observed returns, whose $t$-th column is the vector of returns in period $t$. We assume the mean of $\mathbf{R}$ is zero. The goal is to find a low-rank approximation of $\mathbf{R}$. In particular, we want to find the loadings and factor returns that minimized the total “unexplained” variation of returns, summed across periods and assets, then we would solve the problem
(7.1)
$$ \min_{\mathbf{B}, \mathbf{F}} \| \mathbf{R} - \mathbf{B} \mathbf{F}^T \|_F^2 $$
where $\| \cdot \|_F$ is the Frobenius norm. A matrix of the form $\mathbf{B} \mathbf{F}^T$ above has rank less than or equal to $m$. Conversely, every matrix with rank less than or equal to $m$ can be decomposed as $\mathbf{B} \mathbf{F}^T$ (Exercise 7.1). The problem can be restated as
(7.2)
$$ \min_{\text{rank}(\hat{\mathbf{R}}) \le m} \| \mathbf{R} - \hat{\mathbf{R}} \|_F^2 $$
Here, we have not specified whether the norm is Frobenius. It could be Frobenius, but it could also be any unitarily invariant norm.[1]

We use[2] the Singular Value Decomposition (SVD) $\mathbf{R} = \mathbf{U} \mathbf{S} \mathbf{V}^T$, where $\mathbf{U} \in \mathbb{R}^{N \times N}$ and $\mathbf{V} \in \mathbb{R}^{T \times T}$ are orthonormal matrices of left and right singular vectors. The matrix $\mathbf{S} \in \mathbb{R}^{N \times T}$ is (possibly rectangular) diagonal, and has zero values elsewhere. The solution to Problem (7.2) is given by $\hat{\mathbf{R}} = \mathbf{U}_m \mathbf{S}_m \mathbf{V}_m^T$, where $\mathbf{S}_m$ has the singular values in descending order, with singular values after the $m$-th set to zero. The solution is also due to Eckart and Young (1936). The columns of $\mathbf{U}_m$ are the first $m$ columns of $\mathbf{U}$, and the matrices obtained by taking the first $m$ columns of $\mathbf{U}$ and $\mathbf{V}$. $\mathbf{U}_m \in \mathbb{R}^{N \times m}$ and $\mathbf{V}_m \in \mathbb{R}^{T \times m}$. $\mathbf{S}_m$ is the square matrix obtained by taking the first $m$ columns and $m$ rows of $\mathbf{S}$. Then, the original Problem (7.1) is solved by setting
(7.3) $\hat{\mathbf{B}} = \mathbf{U}_m \mathbf{S}_m^{1/2}$
(7.4) $\hat{\mathbf{F}} = \mathbf{V}_m \mathbf{S}_m^{1/2}$
As noted in earlier chapters, there are equivalent, “rotated” solutions, of the form $\tilde{\mathbf{B}} = \hat{\mathbf{B}} \mathbf{C}^{-1}$, $\tilde{\mathbf{F}} = \hat{\mathbf{F}} \mathbf{C}^T$, for some non-singular $\mathbf{C} \in \mathbb{R}^{m \times m}$. For example, this is also a solution:

(Right Sidebar from Page 272)
(7.5) $\hat{\mathbf{B}} = \mathbf{U}_m \mathbf{S}_m$
(7.6) $\hat{\mathbf{F}} = \mathbf{V}_m^T$
A related problem, with which many readers are acquainted, is Principal Component Analysis (PCA). In this setting, we start with a covariance matrix $\hat{\Sigma} = \frac{1}{T} \mathbf{R} \mathbf{R}^T$. Our goal is to generate a linear combination of the original variables $\mathbf{r}_1^T, \dots, \mathbf{r}_N^T$, i.e., $\mathbf{w}^T \mathbf{r}_t = \sum_{i=1}^N w_i r_{it}$. The vector $\mathbf{w} \in \mathbb{R}^N$ is a set of weights, normalized to have unit Euclidean norm. We want these random observations $\mathbf{w}^T \mathbf{r}_t$ to have the greatest possible variance. With a little work (which we did in previous chapters, or do Exercise 7.2), you can show that this variance is equal to $\mathbf{w}^T \hat{\Sigma} \mathbf{w}$. The problem then can be stated as
$$ \max_{\mathbf{w}} \mathbf{w}^T \hat{\Sigma} \mathbf{w} $$
$$ \text{s.t. } \mathbf{w}^T \mathbf{w} = 1, \|\mathbf{w}\| \le 1 $$
The vector $\mathbf{w}$ is called the first principal component of $\hat{\Sigma}$. You can interpret Section 7.1.1 above as a financial problem. The connection between the approximation problem and the variance maximization of principal components problem is less than trivial to see.[3] The connection between PCA and eigenvalue problems is well known, but it is still useful to highlight it. The Lagrangian of Problem (7.7) is
$$ \mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T \hat{\Sigma} \mathbf{w} - \lambda (\mathbf{w}^T \mathbf{w} - 1) \implies \nabla_w \mathcal{L} = 2\hat{\Sigma}\mathbf{w} - 2\lambda\mathbf{w} $$
, necessary condition for the maximum is that the Lagrangian be zero. This is equal to the eigenvalue equation $\hat{\Sigma}\mathbf{w} = \lambda\mathbf{w}$. From this equation it follows that $\lambda = \mathbf{w}^T \hat{\Sigma} \mathbf{w}$. Therefore, the solution is the eigenvector with the highest associated eigenvalue.

Once this maximum-variance portfolio $\mathbf{w}_1^{(1)}$ has been found, we repeat the process and find another maximum-variance portfolio that is orthogonal to $\mathbf{w}_1^{(1)}$:
$$ \max_{\mathbf{w}} \mathbf{w}^T \hat{\Sigma} \mathbf{w} $$
$$ \text{s.t. } \mathbf{w}^T \mathbf{w} = 1, \mathbf{w}^T \mathbf{w}_1^{(1)} = 0 $$
To see the relationship between PCA and SVD, let us use the uncentered covariance matrix using the SVD decomposition:
(7.8)
$$ \hat{\Sigma} = \frac{1}{T} \mathbf{R} \mathbf{R}^T = \frac{1}{T} \mathbf{U} \mathbf{S} \mathbf{V}^T \mathbf{V} \mathbf{S}^T \mathbf{U}^T = \frac{1}{T} \mathbf{U} \mathbf{S} \mathbf{S}^T \mathbf{U}^T $$
Replace this decomposition of $\hat{\Sigma}$ in the optimization problem, Equation (7.7), and left-multiply by $\mathbf{U}^T \mathbf{w} = \tilde{\mathbf{w}}$. Because the matrix $\mathbf{U}$ is orthonormal, $\|\tilde{\mathbf{w}}\| = \|\mathbf{w}\|$.
The solution is straightforward: $\tilde{\mathbf{w}} = (1, 0, \dots, 0)^T$, and $\mathbf{w}$ equal to the first column of $\mathbf{U}$. If we were to find the first $m$ principal components, we would find that the columns of $\mathbf{U}_m$ solve our problem. These columns, however, are not uniquely identified when some of the eigenvalues are equal. For example you should verify that, if $\lambda_k = \lambda_j$, then any vector
$$ \mathbf{v} = (v_1, v_2, 0, \dots, 0) $$
, with $v_1^2 + v_2^2 = 1$, is indeed a solution. Figure 7.1 gives a geometrical interpretation of this fact.

[Image: Figure 7.1: The eigenvectors associated with identical eigenvalues are not uniquely identified. Shows an ellipse with two possible eigenvector pairs (V1, V2) and (V'1, V'2) for the same eigenvalues.]

We call these vectors interchangeably Principal Components, Eigenvectors, and Eigenportfolios. The variance of the components are the squared singular values of the SVD of the covariance matrix $\hat{\Sigma}$.
Finally, we note that the Optimization Problem (7.7) can be extended to the case of eigenportfolios:
(7.11)
$$ \max_{\mathbf{W} \in \mathbb{R}^{N \times m}} \text{trace}(\mathbf{W}^T \hat{\Sigma} \mathbf{W}) $$
$$ \text{s.t. } \mathbf{W}^T \mathbf{W} = \mathbf{I}_m $$

**7.1.2 Maximum Likelihood Estimation and PCA**

The statistical model was introduced as a norm-minimization problem, but is not directly related to a factor model formulation:
(7.12) $\mathbf{r}_t = \mathbf{B} \mathbf{f}_t + \mathbf{\epsilon}_t$
In fact, if we approximated the covariance matrix with a principal component approximation using the top $m$ eigenvalues, we would obtain a singular covariance matrix, which is highly undesirable.

The goal of this section is to establish a firm connection between spectral methods and the standard factor model. We consider the model above as a starting point. We assume for simplicity that $f_{1t}, \dots, f_{mt}$, the asset idiosyncratic volatilities, are all equal to $\sigma^2$. Furthermore we assume, without loss of generality, that $\Sigma_f = I_m$. This is allowed, because rotational invariance effects on this choice of factor covariance matrix. This is the probability PCA (PPCA) of Bishop (2000).
Under the assumptions $\mathbf{f}_t \sim N(0, \mathbf{I}_m)$ and $\mathbf{\epsilon}_t \sim N(0, \sigma^2 \mathbf{I}_N)$, the return covariance matrix is $\Sigma_R = \mathbf{B} \mathbf{B}^T + \sigma^2 \mathbf{I}_N$. The log-likelihood (times $2/T$) of the observations, conditional on the values of the covariance matrix, is. The log-likelihood function for a zero-mean multivariate normal distribution is (Bishop, 2000; Johnson and Wichern, 2007)
(7.13)
$$ \frac{2}{T} \sum_{t=1}^T \left( -\frac{1}{2} \log |\hat{\Sigma}_R| - \frac{1}{2} \mathbf{r}_t^T \hat{\Sigma}_R^{-1} \mathbf{r}_t \right) = -\log |\hat{\Sigma}_R| - \text{tr}(\hat{\Sigma}_R^{-1} \hat{\Sigma}) + N \log(2\pi) $$
where we denote the scalar product of two matrices
$$ \langle \mathbf{A}, \mathbf{B} \rangle := \text{trace}(\mathbf{A}^T \mathbf{B}) $$
The parameters $\mathbf{B}, \sigma$ can be estimated via maximum likelihood.
(7.14)
$$ \max_{\mathbf{B}, \sigma} -\log |\hat{\Sigma}_R| - \text{tr}(\hat{\Sigma}_R^{-1} \hat{\Sigma}) $$
$$ \text{s.t. } \hat{\Sigma}_R = \mathbf{B} \mathbf{B}^T + \sigma^2 \mathbf{I}_N $$
The solution to this problem is especially simple and intuitive (Tipping and Bishop, 1999). Decompose $\hat{\Sigma} = \mathbf{U} \mathbf{S} \mathbf{U}^T$. Then
(7.15)
$$ \mathbf{B} = \mathbf{U}_m (\mathbf{S}_m^2 - \sigma^2 \mathbf{I}_m)^{1/2} $$
$$ \hat{\sigma}^2 = \bar{\lambda} $$
where $\bar{\lambda}$ is the average of the last $N-m$ eigenvalues of $\hat{\Sigma}$. An alternative rotation of this same model is
(7.16) $\hat{\mathbf{B}} = \mathbf{U}_m$
(7.17) $\hat{\Sigma}_f = \mathbf{S}_m^2 - \bar{\lambda} \mathbf{I}_m$
The model offers several insights. First, it links a probabilistic model of returns to the PCA of the empirical covariance matrix. Second, in the PPCA solution the factor covariance matrix is diagonal and the factor variances are equal to the shrunken empirical variances obtained by PCA. Indeed, the PCA solution can be obtained as an asymptotic result. Consider the limit $N \to \infty$. In this scenario, the idiosyncratic risks are much smaller than the factor risk. In the limit, the formula then simplifies to
(7.18) $\hat{\mathbf{B}} = \mathbf{U}_m$
(7.19) $\hat{\Sigma}_f = \mathbf{S}_m^2$
which is the PCA solution.
We show how PPCA works in a simulated instance. We choose $m=2, T=250$. It is equal to 1000 and 2000 assets, $N=10$. We can simulate returns according to Equation (7.12) for some fixed parameters, say $m=10$.
Figure 7.2 shows the population factor variances (circles) and the shrunken factor variances (triangles). We can see that, when the number of assets and the number of periods is greater, the upward bias of the sample eigenvalues—i.e., of the sample factor variances—is higher. Shrinkage eliminates such bias. However, the shrunken eigenvalues are biased downwards and, additionally, the downside is that the optimal shrinkage parameter $\bar{\lambda}$ is optimal but should not be a constant offset. There are three takeaways from these simulations, which could be confirmed empirically for other choices of the parameters:
*   Sample factor eigenvalues are higher than their population counterparts.
*   Shrinkage helps, but optimal shrinkage may be more complex than a simple offset.
*   Maximum likelihood estimation, which we could solve analytically in this special case, will give in general biased estimates on the factor volatilities.

[Image: Figure 7.2 (a) Probabilities PCA for a universe of 1000 assets, with 10 factors with volatilities 1, 2, ..., 10. Circle-dashed points are the sample factor variances against the true variances; triangle-joined are the shrunken factor variances against the true variances. (b) All parameters are unchanged, with the exception of the number of assets now equal to 2000.]

**7.1.3 Cross-Sectional and Time-Series Regressions via SVD**

A popular approach to PCA is to take the first $m$ principal components of the PCA as factor loadings, and then estimate the factor returns via cross-sectional regression. We are familiar with this approach from Chapter 4. The cross-sectional factor returns are the result of $T$ cross-sectional regressions. We can write the relation as follows:
(7.20)
$$ \hat{\mathbf{f}}_t = (\hat{\mathbf{B}}^T \hat{\mathbf{B}})^{-1} \hat{\mathbf{B}}^T \mathbf{r}_t = (\mathbf{U}_m^T \mathbf{U}_m)^{-1} \mathbf{U}_m^T \mathbf{r}_t $$
The least-squares estimate is
(7.21)
$$ \hat{\mathbf{F}} = (\hat{\mathbf{U}}_m^T \hat{\mathbf{U}}_m)^{-1} \hat{\mathbf{U}}_m^T \mathbf{R} $$
or, since $\hat{\mathbf{U}}_m$ is orthonormal,
(7.22)
$$ \hat{\mathbf{F}} = \hat{\mathbf{U}}_m^T \mathbf{R} = \hat{\mathbf{U}}_m^T \mathbf{U} \mathbf{S} \mathbf{V}^T = \mathbf{S}_m \mathbf{V}_m^T $$
Behold, these are the same factor estimates we computed from the SVD in Section 7.1.1. If we throw away the factor returns beyond the $m$-th, the loadings of the SVD itself allow us to recover them from cross-sectional regressions. Similarly, you can easily prove that, if we only know the estimated factor returns $\hat{\mathbf{F}}$ from Equation (7.22), then we can estimate the loadings using time-series regression of asset returns against these factor returns, and obtain $\hat{\mathbf{B}} = \mathbf{U}_m$. Connor and Korajczyk (1986, 1988) show that this estimator of $\mathbf{B}$ is the only linear combination of the time series of asset returns to the factor returns, and the factor returns are the cross-sectional betas of the asset returns to the loadings.[4] This computational simplification, but also has several applications. It is a useful pedagogical device, since it highlights the connection between time-series and cross-sectional performance attribution in fundamental factor models, and establishes a connection between statistical and fundamental factor models.

---

**7.2 Beyond the Basics**

It is important to understand the behavior of PCA in finite samples, and in settings that are relevant to portfolio managers. There are a few parameters that intuitively should matter to the portfolio manager. The first two are trivial: the number of assets $N$ and the number of factors $m$. In addition, we will perform SVD on a rolling window of observations of width $T$. In Figure 7.3 the cross section of returns are drawn from the same distribution, but different windows of data are used. The number of observations to estimate the parameters.

[Image: Figure 7.3: We estimate the risk model parameters using data in an interval of width $T$.]

Finally, another important quantity is the gap between the $m$-th and the $(m+1)$-th eigenvalues, corresponding to the separation between the smallest variance of a factor and the largest idiosyncratic variance. How do these quantities interact? This question has been at the center of intense research in the past 25 years. PCA, a century-old technique, has witnessed a theory renaissance which is still far from being complete. This section aims to summarize some results, to compare them to simulated scenarios, and finally to administer some practical advice in using PCA.

**7.2.1 The Spiked Covariance Model**

Let $\tilde{\mathbf{R}} \in \mathbb{R}^{N \times T}$, with $T \ge N$, be the sorted eigenvalues of the empirical covariance matrix
(7.23)
$$ \tilde{\Omega}_T = \frac{1}{T} \sum_{t=1}^T \mathbf{r}_t \mathbf{r}_t^T $$
The spiked covariance model posits the following: there is $0 < m < N$ and a positive constant $c$ such that if $N, T \to \infty$:
(7.24)
$$ \lambda_i = \begin{cases} \lambda_i^* > c & \text{for } i \le m \\ c & \text{for } i > m \end{cases} $$
$$ \lambda_i^* = \lim_{T \to \infty} \tilde{\lambda}_i $$
There are $m$ eigenvalues (the “spikes”) that are larger than $c$, and the remaining ones converge to $c$. How does this relate to factor models? Consider the original model specified by Equation (7.12) and choose, like we did in Section 7.1.2, and set $\sigma^2 = c$. A formulation
(7.25)
$$ \lambda_i = \begin{cases} \lambda_i^* > c & \text{for } i \le m \\ c & \text{for } i > m \end{cases} $$
Why should the eigenvalues $\lambda_i$ grow at least linearly in $N$? The first $m$ eigenvalues of $\mathbf{B} \mathbf{B}^T$ are the same as those of $\mathbf{B}^T \mathbf{B}$. To see this, write the SVD decomposition of $\mathbf{B} = \mathbf{U} \mathbf{S} \mathbf{V}^T$ and consider the two matrix products $\mathbf{B} \mathbf{B}^T = \mathbf{U} \mathbf{S}^2 \mathbf{U}^T$ and $\mathbf{B}^T \mathbf{B} = \mathbf{V} \mathbf{S}^2 \mathbf{V}^T$. The two products have the same non-zero eigenvalues. Instead of analyzing the properties of $\mathbf{B} \mathbf{B}^T$, we will work with $\mathbf{B}^T \mathbf{B}$.
A reasonable assumption for $\mathbf{B}$ is that its rows $\mathbf{b}_i$ representing the loadings of a single stock to the factors, are iid samples from a probability distribution $D$, so that $E_D[\mathbf{b}_i \mathbf{b}_i^T] = \frac{1}{N} \sum_{i=1}^N \mathbf{b}_i \mathbf{b}_i^T = \mathbf{I}_m$. We then write
$$ \mathbf{B}^T \mathbf{B} = \sum_{i=1}^N \mathbf{b}_i \mathbf{b}_i^T = N \left( \frac{1}{N} \sum_{i=1}^N \mathbf{b}_i \mathbf{b}_i^T \right) $$
For large values of $N$, the term in parentheses converges to an expectation $E_D[\mathbf{b}_i \mathbf{b}_i^T]$. We denote $\mu_i$ the eigenvalues of this matrix. The eigenvalues of $\mathbf{B}^T \mathbf{B}$ are then in the limit $\approx N \mu_i$ or equal to $N \mu_i$. The eigenvalues of $\mathbf{B} \mathbf{B}^T$ are then $N \mu_1, \dots, N \mu_m, 0, \dots, 0$. This heuristic argument justifies the scaling by $N$. In the spiked covariance model, the eigenvalues (or spikes) eigenvalues separate for the rest (or bulk), and the gap grows linearly in the size of the stock universe.
Let $\tilde{\mathbf{V}}_m$ be the eigenvectors of $\mathbf{B} \mathbf{B}^T$. The spectrum of the covariance matrix is then given by
$$ \mathbf{V}_1, \dots, \mathbf{V}_m, \mathbf{V}_{m+1}, \dots, \mathbf{V}_N, \quad \lambda_1, \dots, \lambda_m, c, \dots, c $$
so a factor model, after rescaling (so that $\tilde{\mathbf{V}}_m^T \tilde{\mathbf{V}}_m = \mathbf{I}_m$) and rotation (so that $\tilde{\mathbf{B}}^T \tilde{\mathbf{B}} = \mathbf{I}_m$), has an associated spiked covariance matrix. We can see how these conditions translate into practice. In Section 7.3, we will see that the $m$ factor-mimicking portfolio is
$$ \mathbf{w}_i = \mathbf{B}(\mathbf{B}^T \mathbf{B})^{-1} \mathbf{e}_i $$
Consider the risk decomposition:
The factor variance is
$$ \mathbf{w}_i^T (\mathbf{B} \mathbf{B}^T) \mathbf{w}_i = \mathbf{e}_i^T (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T \mathbf{B} \mathbf{B}^T \mathbf{B} (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{e}_i = 1 $$
The idiosyncratic variance is
$$ \mathbf{w}_i^T \mathbf{e}_i = \mathbf{e}_i^T (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T \mathbf{e}_i $$
$$ = \mathbf{e}_i^T (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{e}_i $$
$$ \le \|\mathbf{e}_i^T (\mathbf{B}^T \mathbf{B})^{-1}\|_2^2 \|\mathbf{e}_i\|_2^2 $$
$$ \le \lambda_{min}^{-1} (\mathbf{B}^T \mathbf{B}) \|\mathbf{e}_i\|_2^2 $$
$$ \le 1/(CN) $$
since the norm of an orthonormal matrix $\mathbf{V}_m^T \mathbf{e}_i$ is one.
Therefore for large asset universes, i.e., $N \to \infty$, factor-mimicking portfolios have a vanishingly small percentage idiosyncratic variance. They “mimic” the true factor returns well. A different way to state the approximation property is that the idiosyncratic risk “diversifies away” as the number of assets becomes large. This means that “pure factor” portfolios with factor risk are well above their idiosyncratic risk.

**7.2.2 Spectral Limit Behavior of the Spiked Covariance Model**

The first asymptotic limits for PCA were concerned with large samples $T \to \infty$ and fixed $N$. In this case, Anderson (1963) showed that the sample eigenvalues and eigenvectors converge to their population counterparts. However, the case $N \to \infty$ is more relevant, because $N/T = \gamma \in (0, \infty)$ is often in the order of magnitude as the number of variables.
Here, it is useful to taking $N, T \to \infty$. Assume that:
1.  The $m$ eigenvalues of $\mathbf{B}^T \mathbf{B}$ have finite fourth moments;
2.  there are $m$ constants $c_i$, with
    $$ 0 < c_1 < c_2 < \dots < c_m $$
    such that as $N, T \to \infty$
    (7.26)
    $$ \frac{\gamma}{\lambda_i} \to c_i, \quad i=1, \dots, m $$
3.  The remaining $N-m$ eigenvalues are equal to one.
Then the following holds (Shen et al., 2016; Johnstone and Paul, 2018):
1.  When $\lambda_i > 1+\sqrt{\gamma}$,
    Let $\tilde{\lambda}_i$ be the $i$-th sample eigenvalue. Then
    (7.27)
    $$ \tilde{\lambda}_i \to \mu_i := \lambda_i \left(1 + \frac{\gamma}{\lambda_i-1}\right) \quad \text{a.s.} $$
    Because $\mu_i > \lambda_i$ (Equation 7.27), the limit, as $T \to \infty$, this is the same as
    (7.28)
    $$ \tilde{\lambda}_i \to \lambda_i \left(1 + \frac{\gamma}{\lambda_i-1}\right), \quad i=1, \dots, m $$
    The empirical eigenvalues are asymptotically unbiased for large values of $\lambda_i$.
    Let $\tilde{\mathbf{U}}_i$ denote the population (true) eigenvector and $\mathbf{U}_i$ the sample eigenvectors. Then, almost surely,
    (7.29)
    $$ |\langle \tilde{\mathbf{U}}_i, \mathbf{U}_i \rangle| \to \frac{1}{\sqrt{1+\frac{\gamma \lambda_i}{(\lambda_i-1)^2}}}, \quad i \le m $$
2.  When $\lambda_i \le 1+\sqrt{\gamma}$,
    $$ \tilde{\lambda}_i \to (1+\sqrt{\gamma})^2 \quad \text{in probability} $$
    $$ |\langle \tilde{\mathbf{U}}_i, \mathbf{U}_i \rangle| \to 0 \quad \text{a.s.} $$
Even if this strong result only holds asymptotically, it offers a few insights that can be verified experimentally. In addition, there are similar results that extend to the multiple spiked eigenvalue case, albeit with more assumptions. First, let us stress the insights:
1.  Under the spiked model assumptions, the spiked empirical eigenvalues are asymptotically upwardly biased. The bias is higher if $\lambda_i$ is closer to the ground eigenvalue; it becomes smaller when $\lambda_i$ gets bigger. This makes intuitive sense. When $\lambda_i$ is close to one, then the probability that the largest empirical eigenvalue is a “noise” ground eigenvalue becomes non-negligible. This brings us to the next insight.
2.  There is a critical threshold at $1+\sqrt{\gamma}$. For eigenvalues larger than $1+\sqrt{\gamma}$, it is possible to separate the largest eigenvalue from the spectrum. Indeed, the largest sample eigenvalue is further beyond $(1+\sqrt{\gamma})^2$. The separation is clearer still when $\lambda_i$ is large. The larger the first eigenvalue, the better the eigenvector’s collinearity.
Below the threshold, the largest eigenvalue, even if it is larger than $1$, cannot easily be identified from the data. The associated eigenvector is not consistently estimated.
In practice, for many applications, the number of assets in a model is in the interval $[10^3, 10^4]$, and the number of observations ranges between 250 and 1000, so that $\sqrt{\gamma}$ ranges between 1 and 2. This is a useful starting point to reason about thresholding eigenvalues, and their associated eigenvector.

**7.2.3 Optimal Shrinkage of Eigenvalues**

We know that the empirical eigenvalues are biased. This means that, should we evaluate portfolios in the subspace spanned by the spike eigenvectors, the predicted variance of the portfolios will be biased upward by $\tilde{\lambda}_i^2 / \lambda_i^2$. Let $\mathbf{w}$ be a unit-norm vector, and the portfolio be $\mathbf{w}^T \mathbf{U}_m$. Then the true variance is $\mathbf{w}^T \mathbf{S}_m^2 \mathbf{w} = \sum_{i=1}^m w_i^2 \lambda_i^2$, and the predicted variance is $\sum_{i=1}^m w_i^2 \tilde{\lambda}_i^2$. The problem is to estimate $\lambda_i^2$. From Equation (7.27), we could invert $\tilde{\lambda}_i = \lambda_i (1 + \gamma/(\lambda_i-1))$ and solve for $\lambda_i$. This is a quadratic equation, and we could insert $\tilde{\lambda}_i$ for $\lambda_i$ on the RHS. From Equation (7.27),
(7.30)
$$ \ell(\lambda) = \lambda_i - \gamma \frac{\lambda_i}{\lambda_i-1} \ge 1+\sqrt{\gamma} $$
For large values of $\lambda_i$, this is an offset of the empirical eigenvalues, like the one we saw in PCA, Equation (7.17). When we apply this to a diagonal $\mathbf{S}_m^2$ and $\mathbf{B} = \mathbf{U}_m \mathbf{S}_m^{1/2}$, we use the matrix $\mathbf{S}_m^2 - \bar{\lambda} \mathbf{I}_m$ (which is not diagonal). However, this is not necessarily the best choice. The shrinkage of eigenvalues has a long history. Donoho et al. (2018) characterize the optimal shrinking of eigenvalues for a large number of functions. Based on what we learned in Section 7.2.2, we only use a few of the largest eigenvalues. The optimal shrinkage formula (Equation (7.30)) is optimal for large values of $\lambda_i$. For small eigenvalues close to $1+\sqrt{\gamma}$, we shrink them proportionally more than the small values. The Ledoit-Wolf (2004) shrinkage method, which is a very popular method among practitioners: the Ledoit-Wolf shrinkage method starts with finding a matrix of the form
(7.31)
$$ \hat{\Sigma}_R = \rho_1 \hat{\Sigma} + \rho_2 \mathbf{I}_N $$
and identifying $\rho_1$ and $\rho_2$ so that $\hat{\Sigma}_R$ minimizes the distance reduced by the Frobenius norm from $\Sigma_R$.
(7.32)
$$ \min_{\rho_1, \rho_2} \| \Sigma_R - \rho_1 \hat{\Sigma} - \rho_2 \mathbf{I}_N \|_F^2 $$
The space of $m \times m$ matrices is a Hilbert space with scalar product $\langle \mathbf{A}, \mathbf{B} \rangle = \text{trace}(\mathbf{A}^T \mathbf{B})$.
The problem (7.32) is then just a special case of the well-known problem of minimum distance of a subspace to a point in a Hilbert space (Luenberger, 1969, Sec 3.8). They assume iid returns, finite fourth moments, and an asymptotic regime in which $N$ is constant and $T \to \infty$. They find that the optimal solution is of the form
(7.33)
$$ \hat{\Sigma}_R = (1-b^2) \hat{\Sigma} + b^2 \frac{\text{tr}(\hat{\Sigma})}{N} \mathbf{I}_N $$
This solution has many interpretations, aside from the geometric one that follows from the solution to Optimization Problem (7.32). While these interpretations may be of independent interest, I will devote some time to justify why this specific shrinkage is not necessarily optimal in our context. The main issue is that the Frobenius-norm-induced shrinkage will necessarily tend to identify the structure of the model, as shown in the previous chapter (Fan et al., 2008). Secondly, because the regime is $N$ fixed, $T$ diverging is not relevant to applications in which $N > T$ or $N \approx T$. Thirdly, because the constant that multiplies $\mathbf{I}_N$ is the average of the eigenvalues, it may be overly conservative. Lastly, because the eigenvalues of the largest matrix of the form $\lambda_i \mathbf{I}_N - \gamma (\lambda_i-1)^{-1}$ for the leading eigenvalues, this shrinkage model does not match the optimal asymptotic shrinkage of the spiked covariance model.

**7.2.4 EigenShrink: Eigenvalue versus Theory**

We now compare these theoretical results to simulations. We use the same setup as used for the PCA results in Figure 7.2, i.e., $m=10$ factors with standard deviations ranging between 1 and 10, uniformly spaced, and idiosyncratic standard deviations; 250 periods, and either 1000 or 2000 assets. In addition to the case of normal returns, I also consider the case of heavy-tailed returns. Specifically, both factor returns and idiosyncratic returns are t-distributed with five degrees of freedom. This choice is meant to simulate returns that have finite fourth moments, which is a reasonable assumption for daily asset returns.
We simulate 50 instances of each factor model. For each model, we compute the first 20 empirical top eigenvalues, and we shrink them using Equation (7.30) for $\ell(\lambda)$. The simulation (Figure 7.4) shows that the shrinkage function $\ell$ works well for normal returns, but not for heavy-tailed returns. In the latter case, it appears that a shrinkage proportional to the eigenvalue is a common factor. This is a different shrinkage than the ones of Equations (7.30) and (7.31), which are consistent in a constant offsetting term. Combining the empirical observations from simulated data, and theoretical results, it seems at least reasonable to consider a linear shrinkage
$$ \ell(\lambda) = K_1 \lambda - K_2 $$
$$ K_2 \ge \lambda_{min} $$
$$ K_1 \in (0,1] $$
when identifying $\lambda$.

[Image: Figure 7.4 (a) 1000 assets, normally distributed returns; (b) 1000 assets, t-distributed returns; (c) 2000 assets, normally distributed returns; (d) 2000 assets, t-distributed returns. The x-axis denotes the population eigenvalues. The y-axis denotes the shrunken empirical eigenvalues. The dashed line is the line y=x.]

**7.2.5 Choosing the Number of Factors**

In the example above, we assumed that the number of factors was known in advance. This is not the case with applications. An important component of the model definition procedure is the determination of the number of factors.[5] There are some criteria motivated by theoretical models, and others that are the outcome of experiments and trial-and-error by generations of practitioners. The theoretical models provide a lower bound on the number of factors, and we should preserve this number with a high confidence level. But, finding the right factors matters more than finding their exact number. By “right”, we mean the factor loadings with the “best performance”, and by performance, we mean one of many metrics introduced in Chapter 5. Because there are many metrics, and because the choice of the number of factors is a one-size-fits-all criterion. Second consideration, telling the exact number of factors is practice is either very easy or hopelessly hard. Under the assumptions of pervasive factors, we won’t need complex criteria. However, there is a wide gap between the situation eigenvalues and the true eigenvalues of returns. We need to use the eigenvalues that are not too noisy, gradually, and a hard rule is unlikely to choose the must threshold. A final consideration, which is both grounded in theory and in practice, is that one should err on the side of selecting more factors, rather than fewer. The cost of including an extra factor is that the model will choose some spurious noise that will increase true risk, which can result in a reduction in the Sharpe Ratio. The cost of choosing too many factors is a slight decrease in the Sharpe Ratio.
After these qualifications, let us review the most common methods.
*   **Threshold-Based Methods.** For matrices with ground eigenvalues equal to 1, the results of Section 7.2.2 suggest that we should select factor eigenvalues that exceed the threshold $1+\sqrt{\gamma}$, i.e.
    (7.34)
    $$ m = \max \{k | \tilde{\lambda}_k \ge 1+\sqrt{\gamma} \} $$
    An older method is the scree plot. This is the best-known method. In constructing the scree plot, we plot the eigenvalues against their rank. The largest eigenvalues dominate and decrease rapidly. To a value where the eigenvalues are small and decrease gradually, usually almost linearly. The method consists of choosing the last eigenvalue preceding this group. A variant of this method plots the logarithm of the eigenvalue:
    (7.35)
    $$ m = \underset{2 \le k \le K_{max}}{\arg \max} (\tilde{\lambda}_{k-1} - \tilde{\lambda}_k) $$
    (7.36)
    $$ m = \underset{2 \le k \le K_{max}}{\arg \max} (\log \tilde{\lambda}_{k-1} - \log \tilde{\lambda}_k) $$
    where $K_{max}$ is a threshold chosen iteratively (Onatski, 2010).
*   **Penalty-Based Methods.** We began the chapter with the problem of minimizing the square residual error (Equation 7.2). We can select the number of factors by adding a penalty term, and by making it a function of the data and factor numbers.
    (7.37)
    $$ \min_{k: \text{rank}(\hat{\mathbf{R}}) \le k} \| \mathbf{R} - \hat{\mathbf{R}} \|_F^2 + k f(N,T) $$
    (7.38)
    $$ f(N,T) = \frac{N+T}{NT} \log \left( \frac{NT}{N+T} \right) $$

---

**7.3 Real-Life Stylized Behavior of PCA**

We now explore a real-life dataset with the goal of comparing the observed behavior of principal components and eigenvalues to the ideal spiked covariance model. We employ daily stock total returns belonging to the Russell 3000 index for the period 2003-2017. Assets that are included in this index must satisfy some essential requirements. As of 2022, on a designated day in May (“rank day”), Russell evaluates eligible U.S. securities. The stocks must trade on eligible U.S. exchanges. Among them, the 4,000 largest (by market capitalization) become members of the Russell 3000E Index. The smallest 1% of the stocks are excluded. The largest 3,000 stocks become the Russell 3000 Index. The market capitalization must exceed $30M, and the percentage of float (shares traded on exchange) must exceed 5%. The data is adjusted for stock splits, dividends, name changes, governance measures, and corporate structure issues. We use the closing price of each stock. Out of the eligible set, Russell assigns to an index the 3000 stocks with the highest market cap, and effectively changes the composition of the index on the fourth Friday of June. These criteria ensure that the asset characteristics are sufficiently homogeneous. Besides the geography, revenue source, and corporate governance and that the universe of assets is based on daily closing price. Based on stock price and market capitalization.[12]

**7.3.1 Concentration of Eigenvalues**

For our exploration, we consider Principal Components based on three types of returns. First, stock total returns. This is the simplest approach. Secondly, we normalize returns by dividing them by their predicted idiosyncratic volatilities. The benefits of this approach are discussed in the previous sections: the idiosyncratic volatilities of the normalized returns are closer to one. The eigenvalues of the normalized returns should be greater than one. Lastly, we normalize returns to their empirical total volatilities. The rationale for this choice is that we study the properties of the empirical correlation matrix. It has less reasonable than the previous one. Come what may, for our purposes, it is useful to compare the results. The resulting covariance matrix is the sample correlation matrix. We use the procedure in Bai and Ng (2002) and Ahn and Horenstein (2013) to estimate the number of factors. We use one full year of return data, for eight non-overlapping years. When we normalize by idiosyncratic volatilities, we use the data provided by a major U.S. model (Axioma US4). Since the idiosyncratic volatility estimation process takes time, in the chapter, we did not wait any further to show some empirical data. I will show later that the statistical model idiosyncratic volatilities are indeed quite close to those of a commercial model, via this illustrative example, is in fact quite close to a self-contained analysis.

The raw returns are winsorized at daily returns of [-30%, +30%], and the z-scored returns at returns of [-10, +10], i.e., plus or minus ten standard deviations. The idiosyncratic volatilities are winsorized at [0.1, 1.0]. The choice of the first two is not critical. These features are common. The first one is that there is no obvious gap between the variances. The second is that there is a consistent ranking between the spectra of the three covariance matrices. The plot shows the ratio of the variance of lower-order factors to that of the first factor. The standard deviation of the factor returns is higher for the raw returns covariance matrix, followed by the return/volatility covariance matrix, and lastly by the covariance matrix of total returns.

[Image: Figure 7.5: Variances of the eigenvalues (normalized to the variance of the first) for the first 40 factors. Note that the scale of the y-axis is logarithmic.]

This suggests that the first few eigenvectors explain a larger percentage of total variance of the associated covariance matrix. This is confirmed by Figure 7.6. Figure 7.6 plots the cumulative percentage of variance explained by the first 40 factors. For the period ending on June 30, 2008, the first factor explains 30% of the variance of asset returns. For the period ending on June 30, 2011, we need 30 factors for the raw covariance matrix, 15 factors for the z-scored returns, using total volatility, and only 10 factors for the z-scored returns, using idiosyncratic volatility. This suggests that the last choice is preferable. This metric, nonetheless, it suggests that, in this specific instance a model built on a transformed sequence of returns is more parsimonious.

[Image: Figure 7.6: Cumulative percentage of variance described by the first m factors, for difference covariance matrices.]

**7.3.2 Controlling the Turnover of Eigenvectors**

So far, we have focused on the properties of the eigenvalues. Eigenvectors exhibit a distinctive behavior as well. One important property of eigenvectors is their turnover. The turnover for two consecutive portfolios $\mathbf{v}(t-1), \mathbf{v}(t)$ is usually measured as the Gross Market Value (GMV), traded as a percentage of the GMV of the portfolio.
(7.39)
$$ \text{Turnover}_t(\mathbf{v}(t)) := \|\mathbf{v}(t) - \mathbf{v}(t-1)\|_1 $$
Alternatively, one can use as definition the square of the gross national.
(7.40)
$$ \text{Turnover}_t^2(\mathbf{v}(t)) := \|\mathbf{v}(t) - \mathbf{v}(t-1)\|_2^2 $$
There are good reasons for this. The first one is that the squared GMV is a fairly good approximation to the transaction costs associated to trading the factor portfolio. A second one is analytical tractability and an associated geometric intuition for eigenportfolios, recall that $\|\mathbf{v}(t)\|=1$, and that the numerator $\|\mathbf{v}(t) - \mathbf{v}(t-1)\|_2^2$ is
$$ 2(1 - \mathbf{v}^T(t) \mathbf{v}(t-1)) $$
The quadratic turnover is therefore related to the cosine similarity. Low turnover eigenportfolios thus have high cosine similarity.
(7.41)
$$ \text{Turnover}_t(\mathbf{v}(t)) = 2(1 - |S_c(\mathbf{v}(t), \mathbf{v}(t-1))|) $$
In the equation above we use the absolute value of $S_c$ because the eigenvectors are identified modulo the sign of the vector. In other terms, if
$$ S_c(\mathbf{v}(t), \mathbf{v}(t-1)) < 0 $$
, we flip the sign of $\mathbf{v}(t)$ in order to have a lower turnover of eigenvectors.
In Figure 7.7 we show the absolute value of the cosine distances over time for the first eight eigenvectors of our three sequences of covariance matrices computed on raw total returns, raw returns normalized by total volatility, and raw returns normalized by idiosyncratic volatility. In all three cases, we show the total distance between consecutive eigenvectors, computed on the first eight eigenvectors and in Figure 7.7 we show the cumulative returns in the three approaches. The covariance matrix on a given date is computed using the trailing 252 trading days of returns. The number of assets from one day to the next can change slightly. The turnover of the eigenvectors is, on average, low. The charts in Figure 7.7 show that the first eigenvector is associated to an eigenvalue that has a large gap from the second largest eigenvalue (see Figure 7.5). As a result, the PCA procedure has no issue in identifying it and its weights are very stable throughout the estimation period. This is termed a “market” profile. The turnover of the remaining eigenvectors is higher. The turnover of the eigenvectors for the (non-normalized) total returns. There are occasional spikes; for example, there are large spikes occurring on October 3, 2008, and November 20, 2009. The second one is so big that the eigenvectors in consecutive periods are almost orthogonal. What is happening? The market is in turmoil, and immediately before and immediately after the crisis, the eigenvectors change dramatically to a single, stable, but clearly different equilibrium. And this behavior qualitatively repeats across covariance matrices and eigenvectors. This is not an isolated event. In fact, there are more such spikes, but transformed and mild. Another qualitative phenomenon is that, as of February 15, 2008, eigenvalues, standardizing returns seems to reduce turnover and incidence of spikes, more so for idiosyncratic volatility normalization. For example, in the latter case, Figure 7.7 has a cleaner structure and less noise. This is only one of the cases. This lack of eigenvalue separation. When eigenvalues are close, the addition and removal of one or two assets in the estimation universe, is sufficient to affect the number of factors. The instability of the eigenvectors is then due to the instability of the number of factors. Within the change of factors, the eigenvectors change, the subperiod spanned by these eigenvectors may in fact be stable. In Figure 7.7, we show the turnover for the three cases above.[14] When the eigenvalues are small (being just “noise”), for three return vectors, and are smaller for idiosyncratic volatility z-scored returns. This does confirm yet again that statistical models built on normalized return sequences are more stable, suggesting that the eigenvalues of such models are better separated from each other.

[Image: Figure 7.7: Eigenvector turnover for different covariance matrices. Top: total returns; Middle: total returns/total vol; Bottom: total returns/idio vol.]

[Image: Figure 7.8: Distances between columns subspaces of the first eight eigenvectors in consecutive periods. The eigenvectors are generated by PCA on total returns, total returns/total vol, and total returns/idio vol.]

[Image: Figure 7.9: $L_2$ factor turnover for the first four eigenvectors. The eigenvectors are generated by PCA on total returns, total returns/total vol, and total returns/idio vol.]

Aside from the quality of the PCA for different choices of covariance matrices, we are faced with an inescapable issue in statistical models. Except for a few high-order factors and variances, most factors in statistical models suffer from a kind of indeterminacy. In consecutive periods, PCA may give us very different loading vectors, although the subspace spanned by these factors are often in reality very close. This is not a new problem. In the next section, we show that, even if this loadings can change a lot from one period to the next, the covariance matrix does not change. This means that a portfolio’s volatility prediction does not depend on the orientation of the factor loadings, and therefore that any portfolio optimization is also independent of it. This is good news. The bad news is that its factor exposures are also independent of it, which means that it is hard to assess the portfolio volatility, or combined factor volatility of the degenerate factors (i.e., factors with identical volatilities). In integrated fundamental/statistical models the consistency of loadings is not always the first thing that comes to mind. Table 7.1 summarizes the impact of the characteristics of the fundamental factors. Table 7.1 summarizes its specific applications.

**TABLE 7.1**
Summary of impact of High Factor Turnover

|                                       | Impact of High Factor Turnover |
| :------------------------------------ | :----------------------------- |
| Volatility Estimation                 | Not important                  |
| Portfolio Optimization/Hedging        | Not important                  |
| Integrated Stat./Fund. Models         | Not important                  |
| Performance Attribution               | Very high                      |

Essentially only single-factor performance attribution is made irrelevant by eigenvalue quasi-degeneracy. However, single-factor attribution depends on factor turnover. Since most statistical models are affected by rotations, we can always perform a rotation that minimizes the distance between eigenvectors in consecutive periods; this is a per-asset operation. In other words, if we have a sequence of loading matrices $\mathbf{B}_t$, we aim for new “rotated” loadings $\tilde{\mathbf{B}}_t$ that have low turnover.[15]:
(7.42)
$$ \tilde{\mathbf{B}}_{t+1} = \underset{\mathbf{X} \in \mathbb{R}^{N \times m}}{\arg \min} \| \mathbf{B}_t - \mathbf{Y} \mathbf{X} \|_F^2 $$
$$ \text{s.t. } \mathbf{Y} = \mathbf{B}_{t+1} \mathbf{X} $$
$$ \mathbf{X}^T \mathbf{X} = \mathbf{I}_m $$
First, we prove that the objective is equivalent to maximizing $\langle \mathbf{A}, \mathbf{X} \rangle$, with $\mathbf{A} = \mathbf{B}_t^T \mathbf{B}_{t+1}$. This follows from the sequence of identities
$$ \|\mathbf{B}_t - \mathbf{B}_{t+1} \mathbf{X}\|_F^2 = \text{trace}((\mathbf{B}_t - \mathbf{B}_{t+1} \mathbf{X})^T (\mathbf{B}_t - \mathbf{B}_{t+1} \mathbf{X})) $$
$$ = \text{trace}(\mathbf{B}_t^T \mathbf{B}_t) + \text{trace}((\mathbf{B}_{t+1} \mathbf{X})^T (\mathbf{B}_{t+1} \mathbf{X})) - 2 \text{trace}(\mathbf{B}_t^T \mathbf{B}_{t+1} \mathbf{X}) $$
$$ = \text{trace}(\mathbf{B}_t^T \mathbf{B}_t) + \text{trace}(\mathbf{X}^T \mathbf{B}_{t+1}^T \mathbf{B}_{t+1} \mathbf{X}) - 2 \text{trace}(\mathbf{A}\mathbf{X}) $$
The last equality follows from the orthonormality of $\mathbf{B}_{t+1} \mathbf{X}$. Let the SVD of $\mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^T$. Since $\mathbf{X}^T \mathbf{X} = \mathbf{I}_m$, the matrix $\mathbf{X}$ is orthonormal. It follows directly $\mathbf{Y} = \mathbf{V} \mathbf{U}^T \mathbf{X} = \mathbf{V} \mathbf{U}^T$. We replace these expressions in the objective function:
$$ \max_{\mathbf{X}} \text{trace}(\mathbf{A}\mathbf{X}) = \max_{\mathbf{X}} \text{trace}(\mathbf{S}\mathbf{Y}) = \sum_i s_i [\mathbf{a}_i \mathbf{A}^T]_{i,i} $$
which is maximized when $\mathbf{Y} = \mathbf{I}_m$, and so the solution is $\mathbf{X} = \mathbf{V} \mathbf{U}^T$.
Now for the last step. Unitary matrices have all eigenvalues equal to ones and orthogonal eigenvectors $\mathbf{u}_i$. The eigendecomposition of $\mathbf{Y}^T \mathbf{Y} = \mathbf{I}_m$, and the objective function is

---

**7.4 Interpreting Principal Components**

One criticism often leveled against PCA is that its loadings are not easy to interpret. The goal of this chapter is to partially dispel this myth. The output of PCA is interpretable, and in fact sometimes it provides additional non-trivial perspectives for the user.

**7.4.1 The Clustering View**

The first avenue of interpretation is to do no transformation at all. The principal components are uniquely determined up to a change of sign: if $\mathbf{u}_k$ is an eigenvector associated to eigenvalue $\lambda_k$, so is $-\mathbf{u}_k$. We show that their loadings can be interpreted as a clustering membership index (Ding and He, 2004). In order to retain this approach, we partition our $N$ assets into $K$ clusters, each characterized by a set membership $C_k$ and centroids
$$ \mathbf{m}_k = \sum_{i \in C_k} \mathbf{r}_i / |C_k| $$
The number of clusters $K < N$ is unknown. The cluster membership is found by minimizing the sum of squared distances from the centroids:
(7.43)
$$ \min_{C_k, \mathbf{m}_k} \sum_{k=1}^K \sum_{i \in C_k} \| \mathbf{r}_i - \mathbf{m}_k \|^2 $$
(7.44)
$$ \text{s.t. } \mathbf{m}_k = \sum_{i \in C_k} \mathbf{r}_i / |C_k| $$
(7.45)
$$ C_j \cap C_l = \emptyset, \quad i \ne j $$
(7.46)
$$ \bigcup_{k=1}^K C_k = \{1, \dots, N\} $$
We rewrite the objective function as
(7.47)
$$ \sum_{k=1}^K \sum_{i \in C_k} (\mathbf{r}_i - \mathbf{m}_k)^T (\mathbf{r}_i - \mathbf{m}_k) = \sum_{k=1}^K \sum_{i \in C_k} \mathbf{r}_i^T \mathbf{r}_i - \sum_{k=1}^K |C_k| \mathbf{m}_k^T \mathbf{m}_k $$
$$ = \sum_{i=1}^N \mathbf{r}_i^T \mathbf{r}_i - \sum_{k=1}^K \frac{1}{|C_k|} \left( \sum_{i \in C_k} \mathbf{r}_i \right)^T \left( \sum_{j \in C_k} \mathbf{r}_j \right) $$
The first sum is a constant and does not affect the optimization problem. We could represent cluster membership algebraically: let $h_{ik} \in \mathbb{R}$ and define $H_{ik} = 1 / \sqrt{|C_k|}$ if asset $i$ is in cluster $C_k$, zero otherwise. Because an asset must belong to exactly one cluster, $H$ is a column of ones. Define $\mathbf{H} = (h_1, \dots, h_K) \in \mathbb{R}^{N \times K}$.
Let
$$ \mathbf{g} = (\sqrt{|C_1|}\mathbf{m}_1, \dots, \sqrt{|C_K|}\mathbf{m}_K) $$
The condition that each asset belongs to precisely one cluster can be expressed as $\mathbf{H}\mathbf{g} = \mathbf{1}_N$. Therefore, to solve $K$-clustering problem, we need to solve
(7.48)
$$ \max_{\mathbf{H}, \mathbf{g}} \text{trace}(\mathbf{H}^T \mathbf{R} \mathbf{R}^T \mathbf{H}) $$
(7.49)
$$ \text{s.t. } \mathbf{H}_{ik} \in \{0, |C_k|^{-1/2}\} $$
Notice that the columns of $\mathbf{H}$ have unit norm and are orthogonal. Then it is natural to relax the discrete requirements on $\mathbf{H}$ and to solve
(7.50)
$$ \max_{\mathbf{H}} \text{trace}(\mathbf{H}^T \mathbf{R} \mathbf{R}^T \mathbf{H}) $$
This is the same formulation as the optimization version of the uncentered PCA, Equation (7.11). The interpretation of the loadings can then be one of approximate cluster membership. The simplest case is when we cluster on the first principal component. We can separate into two clusters based on some threshold on the loadings. The distribution of the loadings distribution will suggest an appropriate cut-off point. When inspecting multiple eigenvectors, a multivariate clustering algorithm will help identify groups.

**7.4.2 The Regression View**

Another way to interpret the loadings of a statistical model is to represent them as sums of vectors, whose weights are intuitive. Qualitatively we proceed as follows. First, assume meaningful stock characteristics for a given date. We fit a linear regression of the loadings on these characteristics. Then, we interpret the coefficients of the regression. We are left with residuals, which we interpret as the portion of the loadings that is not explained by the characteristics. The statistical model. We regress $\mathbf{B}_{i \cdot}$ on the columns of $\mathbf{G}_i$ and denote the regression coefficients $\beta^{(k)}$. In formulas, $\mathbf{B}_{i \cdot} = \mathbf{G}_i \beta^{(k)} + \mathbf{e}_i^{(k)}$, where $\mathbf{e}_i^{(k)}$ is a vector of residuals. We are interested in the subspace of $\mathbf{G}_i$. If we are not happy with a set of characteristics, then the regression may not explain much of the variance of the loadings. The approach is, of course, not restricted to statistical models; we could apply this regression approach to any pair of risk models, to interpret one based on information contained in the other.
As a (very simplified) example, we consider a model built on U.S. asset returns normalized by idio vol, for the date of July 6, 2017. In order to gain intuition about the loadings, we regress them against style loadings only; we use the same ARS(4) as in these loadings. In Table 7.2 and 7.3, we report only the most significant loadings.

**TABLE 7.2**
Regression coefficients for the first principal component

| Term                  | Estimate | Std.Error | t value | Pr(>|t|) |
| :-------------------- | :------- | :-------- | :------ | :------ |
| (Intercept)           | 1.7E-01  | 1.9E-02   | 9.1E+00 | <2e-16  |
| Dividend Yield        | -2.4E-01 | 1.9E-02   | -1.3E+01| <2e-16  |
| Volatility            | 2.4E-01  | 1.9E-02   | 1.3E+01 | <2e-16  |
| Short Term Momentum   | 2.5E-01  | 1.9E-02   | 1.4E+01 | <2e-16  |
| Long Term Momentum    | 1.3E-01  | 1.9E-02   | 7.0E+00 | 2.8E-12 |
| Earnings Yield        | 3.7E-01  | 1.9E-02   | 1.9E+01 | <2e-16  |

**TABLE 7.3**
Regression coefficients for the second principal component

| Term                  | Estimate | Std.Error | t value | Pr(>|t|) |
| :-------------------- | :------- | :-------- | :------ | :------ |
| (Intercept)           | -2.5E-01 | 1.9E-02   | -1.3E+01| <2e-16  |
| Dividend Yield        | 2.1E-01  | 1.9E-02   | 1.1E+01 | <2e-16  |
| Volatility            | -1.7E-01 | 1.9E-02   | -9.0E+00| <2e-16  |
| Short Term Momentum   | -2.0E-01 | 1.9E-02   | -1.0E+01| <2e-16  |
| Long Term Momentum    | 3.3E-01  | 1.9E-02   | 1.8E+01 | <2e-16  |
| Size                  | -1.8E-01 | 1.9E-02   | -9.3E+01| <2e-16  |

The first principal component is overwhelmingly explained by the market factor. Like the factor attribution of returns, this is usually the case. The most important explanatory variables are Value factor (Dividend Yield), Size, and (with negative coefficient) Short-Term Momentum. The opposite signs for value and momentum are consistent with experience, since the returns of these factors are usually negatively correlated. Size and Earnings Yield loadings are usually positively correlated, the reason being that large caps are likely to pay higher dividends—or dividends at all—than small caps. For this specific date, the correlation is 0.32. The first factor can be interpreted as a “risk-on” factor, where as the second factor can be interpreted as a defensive, or “risk-off” factor.

---

**7.5 Statistical Model Estimation in Practice**

So far, we have only presented the theory of statistical factor models. The next two sections discuss the issues related to its implementation. PCA is usually quite robust (or gentle) that do not have a time dimension. In contrast, we deal with temporal data, and we cannot assume that these data are drawn in each period from the same probability distribution. We will employ the PCA and SVD toolkits, i.e., at most in ways that do not presuppose that the data are approximately stationary. The next two sections discuss two adaptations that are common in practice and compare their performance on historical U.S. equity data.

**7.5.1 Weighted and Two-Stage PCA**

A recurring theme in factor estimation is that weighting observations differently helps. Observations in the distant past are less informative than recent ones; observations of stocks with high idiosyncratic risk should be downweighted compared to those of low-idio stocks. There are therefore two basic transformations that we can apply to the raw return matrix. The first one is time-weighting. We assign to each observation a weight $w_t$ that is a decreasing function of time. Let $W_t = \text{diag}(w_1, \dots, w_T)$ be a diagonal matrix. The weights could be, for example, exponential weights
$$ W_{t,t} = \kappa \exp(-t/\tau) $$
; the positive constant $\kappa$ is such that the squared diagonal terms sum to $T$. Then the time-weighted empirical uncentered covariance matrix is
(7.51) $ \tilde{\Omega}_R = \frac{1}{T} \mathbf{R} W_t \mathbf{R}^T $
This is the same as first transforming the returns $\tilde{\mathbf{R}} = \mathbf{R} W_t^{1/2}$, and then computing the empirical covariance matrix, Equation (7.8), of the transformed returns. In practice, we would not compute the covariance matrix and then perform the PCA, but rather perform the SVD on $\tilde{\mathbf{R}}$, which would be computationally less expensive and give us the same results.
A different type of transformation is cross-sectional reweighting. In Chapter 4 we used it as optimal to returns by the idiosyncratic volatility, at least as a proxy. As with time-weighting, we can apply the transformation first. Take the first $m$ principal components (say, $m=5$) and compute the idiosyncratic returns $\mathbf{E} = \mathbf{R} - \mathbf{B} \mathbf{F}^T$. A case we also consider is $m=0$, in which case $\mathbf{E}=\mathbf{R}$. Define the cross-sectional weights $W_c = \text{diag}(\sigma_1^{-1}, \dots, \sigma_N^{-1})$.
The asset-level reweighted covariance matrix is
(7.52)
$$ \hat{\Sigma} = W_c \mathbf{R} \mathbf{R}^T W_c $$
One can then perform a second-stage factor model on the reweighted covariance matrix.
(7.53)
$$ \tilde{\Sigma} \approx U_m S_m^2 U_m^T + I_N $$
Finally, pre- and post-multiply by the idiosyncratic weighting matrices $W_c^{-1}$.
We employ the steps above in the following process. We use two time-series reweightings with half-life $T_{1/2}$ of 63 for “fast” and $T_{1/2}$ of 252 for “slow”. An empirical insight in asset return data is that volatilities and correlations change over different timescales. Volatilities change rapidly, in fact they may change drastically over the course of a few days. The ratio between volatility during a quiet period and volatility during a quiet period. On the other side, pairwise correlations are quite stable. Even in the presence of large market stresses, these correlations marginally increase in absolute value. This suggests that we separate volatilities and correlations. Therefore, in the second stage we use a short half-life to estimate the idiosyncratic volatilities. In the second stage we use a longer half-life to estimate the factor structure of correlations.

**Procedure 7.1: Statistical model estimation**
1.  Inputs: $\mathbf{R} \in \mathbb{R}^{N \times T}$, $T_{1/2}^{(1)}, T_{1/2}^{(2)}$, $0 < p < N$
    $m > 0$
2.  Time-Series Reweighting
    $W_{t,t}^{(1)} := \kappa \text{diag}(\exp(-T/T_{1/2}^{(1)}), \dots, \exp(-1/T_{1/2}^{(1)}))$
    $\tilde{\mathbf{R}} = \mathbf{R} W_t^{(1)}$
3.  First Stage PCA: $\tilde{\mathbf{R}} = \mathbf{U} \mathbf{S} \mathbf{V}^T$
    Idio Proxy Estimation:
    $\mathbf{E}_t = \mathbf{U}_{m+1 \dots p} \mathbf{S}_{m+1 \dots p} \mathbf{V}_{m+1 \dots p}^T$ (truncated SVD)
    $\hat{\sigma}_i^2 = \frac{1}{T} \sum_{t=1}^T E_{it}^2$ (idio var proxies)
4.  $W_c = \text{diag}(\hat{\sigma}_1^{-1}, \dots, \hat{\sigma}_N^{-1})$
    Idio Reweighting:
    $\mathbf{R}_{idio} := W_c \mathbf{R}$
    $W_{t,t}^{(2)} := \kappa \text{diag}(\exp(-T/T_{1/2}^{(2)}), \dots, \exp(-1/T_{1/2}^{(2)}))$
    $\tilde{\mathbf{R}}_{idio} = \mathbf{R}_{idio} W_t^{(2)}$
5.  Second Stage PCA: $\tilde{\mathbf{R}}_{idio} = \mathbf{U}_{idio} \mathbf{S}_{idio} \mathbf{V}_{idio}^T$
    where: $\mathbf{F} = (\mathbf{B}_{idio}^T \mathbf{W}_c^{-1} \mathbf{B}_{idio})^{-1} \mathbf{B}_{idio}^T \mathbf{W}_c^{-1} \mathbf{R}_{idio} \mathbf{W}_t^{(2)} \mathbf{F}_{idio}$
    (7.54) $\mathbf{B}_{idio} = \mathbf{U}_{idio, m}$
    (7.55) $\mathbf{F}_{idio} = \mathbf{S}_{idio, m} \mathbf{V}_{idio, m}^T$
    We need to address some outstanding problems:
    1.  Sign indeterminacy of eigenvectors.
    2.  Time-varying universe of assets.
    3.  Imputation of loadings for non-estimation universe assets.
    4.  Imputation of missing values for new or temporarily non-traded assets.
    We tackle them in order.
    Sign indeterminacy of eigenvectors. Let us begin with a simple observation: the sign of an eigenvector is arbitrary. If $\mathbf{u}$ is an eigenvector of $\Sigma$ with associated eigenvalue $\lambda$, then so is $-\mathbf{u}$. When we compare the SVD for adjacent periods, we will remove observations, which may lead to a sign flip in the factors. It is important therefore that the loadings of an eigenvector (or factor) in adjacent periods is positive. Aside from the straightforward re-alignment exercise, the turnover of eigenvectors is important in two respects. First, because I observe that
    $$ S_c(\mathbf{u}^T(t), \mathbf{u}^T(t+1)) \approx 0 $$
    It is difficult to determine the sign of consecutive eigenvalues.
    As a result, it is difficult to determine the sign of the factor return $f_t(t)$. This is not a problem for volatility estimation, but it is for any statistical factor model where the sign of the factor return $f_t(t)$ has to be consistent. The second consideration is that very high turnover results in factor-mimicking portfolios with very high turnover as well, and is therefore a factor that is very difficult to trade, either for hedging or for alpha generation.
    Non-estimation universe. Similarly to fundamental models, statistical models are estimated on a predetermined set of assets. The rationale for the choice of such a universe is the same as for fundamental models. Assets in the estimation universe should represent most of the market capitalization of the total strategy. They should be sufficiently liquid to be considered tradable; and, ideally, they should be sufficiently traded to ensure good price discovery and therefore reliable return calculations. Assets enter and leave the universe over time. The number of assets in the estimation universe may be constant or variable. The treatment of additions to the index may be unreliable because the asset was illiquid, or be missing altogether. We can still opt to keep these recent additions, provided that their returns are well defined, or alternatively we can use the assets of the estimation universe of all the common stocks used for the regression. If the universe turnover is not too high, we will still have a sufficiently broad panel of assets. It is preferable to employ an estimation universe that has the lowest possible turnover, and it is important to use a consistent procedure to select the assets to include in the return matrix $\tilde{\mathbf{R}}$.
    Imputing loadings for non-estimation universe assets. There are assets that are not in the estimation universe, but that have complete returns. They do not have factor loadings. We can impute loadings by performing a cross-sectional regression of asset returns on the factor returns. This approach is justified by the results in Section 7.1.3: we can recover loadings from time-series regression, provided that the factor returns are obtained using the estimation universe are close to the true factor returns.
    Imputation of missing values for new or temporarily non-traded assets. Some assets do not have sufficient return history to regress their loadings: examples are newly listed assets (IPOs, ADRs), or assets that were suspended from trading for a long period of time, or had trading volumes suspended over too low to be considered reliable. We can impute its loadings to use additional characteristics of the asset to estimate its loadings. The approach is similar to the one we presented in Section 7.4.2 on the interpretation of loadings using regression. In this case, however, our quantity we are interested in is the loading to impute. We use the style characteristics (Momentum, Value, Volatility). All we have is knowledge of the industry and country of the asset. We regress observed loadings against these two characteristics, and predict the missing loading. It is common practice to remove predicted values that are too small. We will cover a rationale for this practice in later sections devoted to hedging.

3.  Second Stage Factor Model: $\mathbf{Y}_t = \mathbf{U}_{idio} \mathbf{F}_{idio,t} + \mathbf{\epsilon}_t$
    where: $\mathbf{F}_{idio,t} \sim N(0, \text{diag}(\lambda_1^{(idio)}, \dots, \lambda_m^{(idio)}))$
    $\bar{\lambda} = \frac{1}{N-m} \sum_{i=m+1}^N \lambda_i^{(idio)}$
    $\mathbf{\epsilon}_t \sim N(0, \bar{\lambda} \mathbf{I}_N)$
4.  Output: Final Factor Model $\mathbf{Y}_t = \mathbf{B} \mathbf{F}_t + \mathbf{\epsilon}_t$
    where $\mathbf{B} = W_c^{-1} \mathbf{U}_{idio,m}$
    $\mathbf{F}_t \sim N(0, \text{diag}(\lambda_1^{(idio)}, \dots, \lambda_m^{(idio)}))$
    $\mathbf{\epsilon}_t \sim N(0, \bar{\lambda} W_c^{-2})$
This procedure is flexible enough to include several PCA-related procedures as special cases, and to serve as a basis for further experimentation. Some examples:
*   When $m=0$, then idio reweighting becomes a z-scoring, so that the second-stage PCA is effectively applied to the correlation matrix.
*   The optimal use of Equation (7.52) in time is obtained in the single factor model step.
*   It is straightforward to use different shrinkage methods in the second-stage factor model step.
*   In the second-stage factor model step, we use the PPCA results of Section 7.1.2. We can also use these results to estimate their idiosyncratic covariance matrix, and then their covariance, so that their principles. However, we could replace this with a different estimation procedure, like Maximum Likelihood.

**7.5.2 Implementing Statistical Models in Production**

It is not sufficient to have a procedure that estimates the loadings and the covariance matrix at a point in time. In our applications, factor models are dynamic. At time $t$, we have an estimation universe of stocks, and we use return data up to $T_{max}$ periods in the past. We apply the two-stage PCA using return data between $t-T_{max}$ and $t$, to obtain:
*   Loadings $\mathbf{B}_t$. This is the output loadings matrix.
*   Factor returns and estimate returns at time $t$.
(7.55)

---

**7.6 Appendix**

**7.6.1 Exercises and Extensions to PCA**

**Exercise 7.1 (Low-Rank Factorization):**
Prove that a matrix $A \in \mathbb{R}^{m \times n}$ of rank $r \le \min(m,n)$ if and only if it can be decomposed into the product of two matrices $B \in \mathbb{R}^{m \times r}$ and $C \in \mathbb{R}^{r \times n}$.

**Exercise 7.2 (PCA Solution):**
Prove that the solution $\mathbf{w}^*$ to Problems (7.7) and (7.10) is unique and that constraint $\|\mathbf{w}^*\| \le 1$ is always binding, i.e., $\|\mathbf{w}^*\|=1$.

**Exercise 7.3 (Alternative PCA Formulation):**
Prove that the optimization (7.11) gives the same solution as finding the first $m$ eigenvectors of $\Sigma$, and as finding iteratively $k=1, \dots, m$ vectors $\mathbf{w}_k$ that maximize $\mathbf{w}_k^T \Sigma \mathbf{w}_k$ orthogonal to the first $k-1$ vectors, $\mathbf{w}_1, \dots, \mathbf{w}_{k-1}$.

**Exercise 7.4 (Covariance Matrix of a Linear Transformation):**
Prove that if the random vector $\mathbf{x}$ taking values in $\mathbb{R}^n$ has covariance matrix $\Sigma_x$, and if $A \in \mathbb{R}^{m \times n}$, then the random vector $A\mathbf{x}$ has covariance matrix $A \Sigma_x A^T$.

**Exercise 7.5 (A Simple Spiked Matrix):**
Let $B \in \mathbb{R}^{N \times m}$ be an $m$-rank matrix. Prove that the first $m$ eigenvalues of $BB^T + \sigma^2 I_N$ are greater than $\sigma^2$.

**Exercise 7.6:**
Solve the optimization problem Equation (7.1).

**Exercise 7.7 (The Power Method):**
A simple (the simplest) algorithm for computing the largest eigenvalue of a symmetric $A \in \mathbb{R}^{N \times N}$ is the power method.
1.  Start with a unit-norm $\mathbf{x}_0 \in \mathbb{R}^N$, chosen randomly (e.g., sampling from the standard normal distribution, then normalize it).
2.  Iterate: $\mathbf{x}_{k+1} = A \mathbf{x}_k / \|A \mathbf{x}_k\|$.
3.  After the vector converges (say, $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| < \epsilon_{tol}$), approximate the top eigenvector, and $\tilde{\lambda}_N$ the top eigenvalue.
    (Hint:
    $$ \tilde{\lambda}_N = \mathbf{x}_k^T A \mathbf{x}_k = \sum_{i=1}^N \lambda_i (\mathbf{x}_k^T \mathbf{u}_i)^2 $$
    )
Prove the convergence and correctness of the power method. (Hint:
$$ \mathbf{x}_k = \sum_{i=1}^N c_i \mathbf{u}_i \implies A^k \mathbf{x}_0 = \sum_{i=1}^N c_i \lambda_i^k \mathbf{u}_i $$
)

**Exercise 7.8 (Iterations for the SVD):**
A simple (the simplest?) algorithm for computing the largest singular value of a matrix $R \in \mathbb{R}^{N \times T}$ is the following:
1.  Start with $v_0 \in \mathbb{R}^T$. Chosen at random (e.g., sample the coordinates from a standard normal distribution).
2.  Iterate
    (7.56) (i) $u_{k+1} = R v_k$
    (7.57) (ii) $v_{k+1} = R^T u_{k+1}$
3.  After the vectors converge, $s_N = \|u_{k+1}\|$, approximates the highest left eigenvector, $v_{k+1}$ the higher right eigenvector, and $s_N^2 / \|v_{k+1}\|^2$, the top singular value.
Prove the convergence and correctness of the algorithm. (Hint: power method.)
(Hint: How would you extend to find all the eigenvalues of $R^T R$? The same way as you would extend the power method.)

**Exercise 7.9 (Time-Series Regression from the SVD):**
Let $R = U S V^T$, and set $F = V_m S_m$. The vector $\hat{\beta}$ of regression coefficients of the time series of the $n$-th factor return is the $n$-th row of $F$. Prove that the least-squares regression coefficient of the time series of returns on $U_m$ is $U_m$.

**Exercise 7.10 (QR Iterative Algorithm):**
Let $A \in \mathbb{R}^{N \times N}$ be a time series of returns drawn from a common distribution with covariance matrix $\Sigma$. Prove that the following algorithm converges to the first $k$ eigenvalues of $\Sigma$.
1.  Set $j=0$. Choose $X_0 \in \mathbb{R}^{N \times k}$, uniformly at random.
2.  Choose column $q_j$ uniformly at random between 1 and $N$.
3.  Update the direction
    (7.58)
    $$ \mathbf{v}_{j+1} = \mathbf{v}_j + \eta_j (I - \mathbf{v}_j \mathbf{v}_j^T) (\mathbf{r}_j(i) - \mathbf{v}_j^T \mathbf{r}_j(i) \mathbf{v}_j) $$
    (7.59)
    $$ \mathbf{v}_{j+1} \leftarrow \frac{\mathbf{v}_{j+1}}{\|\mathbf{v}_{j+1}\|} $$
4.  Set $j \leftarrow j+1$. If $\|\mathbf{v}_{j+1} - \mathbf{v}_j\| < \epsilon_{tol}$, then stop. Otherwise go to Step 2.
Solution (sketch): Let $R_t \in \mathbb{R}^N$ and $X \in \mathbb{R}^{N \times k}$ be random matrices taking values in $\mathbb{R}^{N \times k}$. Let $X_0$ be one of $T$ values: $X_0^T X_0 = I_k$, with equal probability $1/T$. One can interpret the product $Y^T X_0 R_t^T$ on the expectation $E[Y^T X_0]$. The first $k$ eigenvalues of $E[Y^T X_0 R_t^T]$ are
(7.60)
$$ \lambda_i(E[Y^T X_0 R_t^T]) = \frac{\langle \mathbf{v}_i^* | \mathbf{v}_i^* \rangle}{\langle \mathbf{v}_i^* | \mathbf{v}_i^* \rangle} $$
We can apply (stochastic) gradient algorithm to the maximum search.
The derivation $f(\mathbf{v}) = \mathbf{v}^T E[R R^T] \mathbf{v} - \lambda (\mathbf{v}^T \mathbf{v} - 1)$
The derivative $\nabla f(\mathbf{v}) = 2 E[R R^T] \mathbf{v} - 2 \lambda \mathbf{v}$

**Exercise 7.11 (Distance Between Subspaces):**
Let $A, B \in \mathbb{R}^{N \times k}$ be orthonormal matrices. If the two column subspaces are “similar”, then the $k$-th principal angle between the column subspace of $A$ is well approximated by some unit-norm vector in the column subspace of $B$. Define similarity between the two subspaces as
(7.61)
$$ S(A,B) = \frac{1}{k} \max_{\substack{X \in \mathbb{R}^{k \times k} \\ \|X\|_F=1}} \|AX - BY\|_F^2 $$
Prove that $S(A,B) = 1 - \sigma_1 - \sigma_k$, where $\sigma_1 \ge \dots \ge \sigma_k$ are the $k$ first singular values of $A^T B$.
(Hint: Use the triangle inequality.)

**Exercise 7.12 (Angle Between Subspaces):**
Let $A, B \in \mathbb{R}^{N \times k}$ be orthonormal matrices. Let the least cosine distance between subspaces be the cosine of the smallest achievable angle between two vectors, one belonging to the column subspace of $A$, the other belonging to the column subspace of $B$.
Prove that
$$ S_c(A,B) = \sigma_k(A^T B) $$
where $\sigma_k(A^T B)$ is the last singular value of $A^T B$.

**Exercise 7.13 (Covariance Matrix Degenerate Eigenvalues):**
Consider a risk model with the following structure. Its loading matrix $B$ has the form $B = DU$, where $D \in \mathbb{R}^{N \times N}$ is diagonal positive-definite and $U \in \mathbb{R}^{N \times m}$ is such that $U^T U = I_m$, and its factor covariance matrix is proportional to $I_m$, $\Sigma_f = \lambda_f I_m$.
1.  Prove that we can replace $U$ with an “equivalent” $\tilde{U} = UG$ spanning the same subspace, the estimated covariance matrix does not change.
2.  Extend the result to the case where $\Sigma_f$ is still diagonal, but with the first $m$ variances being greater than the rest
    $$ \lambda_1 > \dots > \lambda_m > \lambda_{m+1} = \dots = \lambda_K $$
    and with
    $$ \tilde{U}_{:,1:p} = U_{:,1:p} $$
    $$ \tilde{U}_{:,(p+1):m} = I_{m-p} $$

**7.6.2 Asymptotic Properties of PCA**

This is a summary of the asymptotic properties of PCA in the regime where the number of variables $N$ is constant and the number of observations $T$ goes to infinity. We use $T$ observations of $N$ variables, $r_1, \dots, r_T$. We assume that the observations are iid from a distribution with covariance matrix $\Sigma$. We denote the $N$ eigenvalues and eigenvectors of the covariance matrix. Decompose the empirical and true covariance matrices into their eigenvalues and eigenvectors:
$$ \hat{\Sigma}_T = \frac{1}{T} \sum_{t=1}^T \mathbf{r}_t \mathbf{r}_t^T $$
By the Strong Law of Large Numbers, $\hat{\Sigma}_T \to \Sigma$ almost surely. Anderson (1963) proves a Central Limit Theorem for the eigenvalues of the covariance matrix. Decompose the empirical and true covariance matrices into their eigenvalues and eigenvectors:
$$ \hat{\Sigma}_T = \hat{U} \hat{S} \hat{U}^T $$
$$ \Sigma = U S U^T $$
with
$$ \lambda_1 > \lambda_2 > \dots > \lambda_N > 0 $$
, all eigenvalues are assumed to be distinct. Anderson proves that, as $T \to \infty$:
(7.62)
$$ \sqrt{T}(\hat{\lambda} - \lambda) \sim N(0, 2\lambda^2) $$
(7.63)
$$ \sqrt{T}(\hat{\mathbf{u}}_i - \mathbf{u}_i) \sim N(0, E_i) $$
(7.64)
$$ E_i = U \begin{pmatrix} 0 & & & \\ & \ddots & & \\ & & (\lambda_i - \lambda_j)^{-2} & \\ & & & \ddots \end{pmatrix} U^T $$
where the $j$-th row has all zeros. Therefore,
1.  The standard error on $\lambda_i$ is $\sqrt{2}\lambda_i / \sqrt{T}$.
2.  The standard error on the principal components, defined as $\sqrt{E[\|\hat{\mathbf{u}}_i - \mathbf{u}_i\|^2]}$, is
    (7.65)
    $$ \frac{1}{\sqrt{T}} \sqrt{\sum_{j \ne i} \frac{\lambda_i \lambda_j}{(\lambda_i - \lambda_j)^2}} $$
    The relative error depends on the separation between eigenvalues.

**The Takeaways**

*   Statistical factor models estimate both factor returns and exposures using return data only, without relying on firm characteristics or macroeconomic data.
*   Advantages: Complementarity to fundamental models, useful when data is sparse, captures short-horizon and multi-asset models.
*   Disadvantages: Interpretability of statistical factors is generally low. PCA identifies a factor model by minimizing the sum of squared residuals in the factor model.
*   Large eigenvalues of the empirical covariance matrix are biased upwards, and are separated in magnitude by a bulk of smaller eigenvalues. This is the spiked covariance model.
*   The sample factor eigenvalues are larger than the true ones, and should be shrinked. Statistical factors can be interpreted. PCA loadings admit a clustering interpretation and a regression interpretation.
*   Statistical models should be rotated in such periods in order to reduce turnover.

**Notes**

1.  [1] These are matrix norms that are invariant for left- and right-multiplication by orthonormal matrices: $\|A\| = \|UAV^T\|$. Special, Frobenius, and nuclear norms.
2.  [2] This problem was solved by Eckart and Young (1936) and generalized by Mirsky (1960). Standard references for PCA are (Johnson and Wichern (2007), Jolliffe (2002), Hastie, Tibshirani, Friedman (2009)). Yao et al. (2015), Jolliffe and Cadima (2016). PCA is also covered in any good book on multivariate statistics (e.g., Mardia, Kent, Bibby (1979)), machine learning (e.g., Bishop (2006), Murphy (2012)). Connor, Goldberg, Korajczyk (2010) is dedicated to the interpretation of SVD, PCA, non-negative matrix factorization and its applications.
3.  [3] This is the full SVD, as opposed to the reduced SVD, see Section 4.7.4.
4.  [4] You can prove (c) of this Solved Exercise 7.2.
5.  [5] In the statistical literature, the analysis of this model begins with Johnstone (2001). In a seminal paper, Chamberlain and Rothschild (1983) impose structure on the covariance matrix by assuming that the returns are generated by an $m$-factor model, with $m \ll N$, and that the idiosyncratic risk is $O(N)$. In our model, in which the diversifiable risk goes to zero.
6.  [6] We have only touched briefly on the asymptotic limit of the spiked model in Section 7.2.2 to give a taste of what happens and give a basis for heuristics. Several papers have characterized the behavior between the model. The first and most famous result is the Marcenko-Pastur (1967) law, which describes the limiting distribution of the eigenvalues of the sample covariance matrix. Bai and Yin (1993) and Bai (1999) extended these results. Baik and Silverstein (2006) and Paul (2007); El Karoui (2008); Mesot et al. (2008); Benaych-Georges and Nadakuditi (2011); Shen et al. (2016); Wang and Fan (2017); Wang and Yao (2017). A survey is Johnstone and Paul (2018). Recent advances are Bun et al. (2017) and Bouchaud and Potters (2019). The line of research concerned with properties of the spectrum in the regime $N/T \approx \gamma > 1$ “begins perhaps with Johnstone (2001)”.
7.  [7] As in Section 7.1.2, $\mathbf{U}_m$ is the submatrix of $\mathbf{U}$ obtained by taking the first $m$ columns.
8.  [8] From the very first result on biased asymptotic estimators, a reader may wonder about shrinkage methods. There is an extensive literature on factor model shrinkage. Standard references are Ledoit and Wolf (2003a,b, 2004) on linear shrinkage, and more recent work on non-linear shrinkage by the same authors (Ledoit and Wolf, 2012, 2015, 2017a,b, 2020). They derive optimal shrinkage functions for a large set of loss functions.
9.  [9] For a relatively old survey on methods to select the number of factors, see Bai and Ng (2002). For a more modern survey is Fan et al. (2013). The scree plot is due to Cattell (1966). The penalty method is due to Akaike (1974). The scree is the debris that form at the base of a cliff.
10. [10] An American Depositary Receipt (ADR) is a foreign company that is listed on a U.S. stock exchange, which also offers shares in U.S. exchange. A Global Depositary Receipt (GDR) is similar to an ADR, but is listed on exchanges in one or more countries outside of their primary market.
11. [11] Note, however, that Russell does not screen stocks based on trading volume, and that the smallest-capitalization companies in Kipoo and Kosso (2013) are often illiquid.
12. [12] For a definition of subspace similarity, see Exercise 7.12.
13. [13] If you are not convinced, or this statement does not seem obvious, this is a good time to solve Exercise 7.11.
14. [14] Note that this problem is closely related to Wahba’s Problem; see Wahba (1990).
15. [15] The two-step procedure for reweighting the PCA is relatively common; Bai and Ng (2002) reweights using idio volatilities, and Bollerslev (1990) using total volatilities.
   

Okay, here is the Markdown compilation for Chapter 8.

```markdown
Everand

