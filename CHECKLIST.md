## Phase 1: Foundational Setup & Core Components
- [x] **Project Initialization**
- [x] **Core Object Skeletons**
- [x] **Utility Module (`utils`) Initial Population Plan**

## Phase 2: Chapter-by-Chapter Implementation
### Chapter 1: The Map and the Territory
- [x] Task B.1: Define Enumerations for Financial Concepts
- [x] Task B.2: Define a Simple Class for Market Participants (Optional, for structure)
- [x] Task B.3: Update `__init__.py` files
- [x] Create example script `examples/chapter_1_concepts_example.py`
- [x] Create unit tests in `tests/test_common_concepts.py`

### Chapter 2: Univariate Returns
- [x] Task B.1: Implement Basic Return Calculations
- [x] Task B.2: Implement Generic Kalman Filter (Appendix 2.4.1)
- [x] Task B.3: Implement Roll Model Price Estimation (using Kalman Filter)
- [x] Task B.4: Implement GARCH(1,1) Model
- [x] Task B.5: Implement Realized Volatility Calculation
- [x] Task B.6: Implement EWMA and State-Space Variance Models
- [x] Create example script `examples/chapter_2_returns_vol_example.py`
- [x] Create unit tests in `tests/test_returns_analysis.py`, `tests/test_volatility_models.py`, `tests/test_utils.py`

### Chapter 3: Interlude: What Is Performance?
- [x] Task B.1: Implement Core Performance Metrics
- [x] Task B.2: Implement Sharpe Ratio Standard Error and Adjustments
- [x] Task B.3: Update `__init__.py` files
- [x] Create example script `examples/chapter_3_performance_example.py`
- [x] Create unit tests in `tests/test_performance_metrics.py`

### Chapter 4: Linear Models of Returns
- [x] Task B.1: Enhance `FactorModelBase` Class
- [ ] Task B.2: Implement Alpha Decomposition
- [ ] Task B.3: Implement Factor Model Transformations
- [x] Task B.4: Implement Basic Linear Regression Utilities (Appendix 4.7.1)
- [ ] Task B.5: Update `__init__.py` files (core and utils need update for FactorModelBase and regression)
- [ ] Create example script `examples/chapter_4_linear_models_example.py`
- [ ] Create unit tests in `tests/test_core_factor_model.py`, `tests/test_factor_models_utils.py`, `tests/test_utils_regression.py`

### Chapter 5: Evaluating Risk
- [x] Task B.1: Implement Robust Loss Functions for Volatility
- [x] Task B.2: Implement Multivariate Covariance Evaluation Procedures
- [x] Task B.3: Implement Ancillary Tests
- [x] Task B.4: Update `__init__.py` files
- [x] Create example script `examples/chapter_5_risk_evaluation_example.py`
- [x] Create unit tests in `tests/test_risk_evaluation.py`

### Chapter 6: Fundamental Factor Models
- [x] Task B.1: Implement Winsorization
- [ ] Task B.2: Create `FundamentalFactorModel` Class Structure
- [ ] Task B.3: Implement Cross-Sectional Regression Step for `FundamentalFactorModel.fit`
- [ ] Task B.4: Implement Factor Covariance (`Ω_f`) Estimation for `FundamentalFactorModel.fit`
- [ ] Task B.5: Implement Idiosyncratic Variance (`Ω_ε` diagonal) Estimation for `FundamentalFactorModel.fit`
- [ ] Task B.6: Implement Currency Hedging Utilities
- [ ] Task B.7: Update `__init__.py` files
- [ ] Create example script `examples/chapter_6_fundamental_model_example.py`
- [ ] Create unit tests in `tests/test_utils_preprocessing.py`, `tests/test_factor_models_fundamental.py`, `tests/test_returns_analysis_currency.py`

### Chapter 7: Statistical Factor Models
- [ ] Task B.1: Create `StatisticalFactorModel` Class Structure
- [ ] Task B.2: Implement PCA-based Factor Estimation for `StatisticalFactorModel.fit`
- [ ] Task B.3: Implement Probabilistic PCA (PPCA) Estimation for `StatisticalFactorModel.fit`
- [ ] Task B.4: Implement Eigenvalue Shrinkage Utilities and Number of Factors Selection (Partially Done - Shrinkage exists, Num Factors missing)
- [x] Task B.5: Implement Eigenvector Alignment (Turnover Control)
- [ ] Task B.6: Update `__init__.py` files.
- [ ] Create example script `examples/chapter_7_statistical_model_example.py`
- [ ] Create unit tests in `tests/test_factor_models_statistical.py`, `tests/test_utils_matrix_ops.py`

### Chapter 8: Evaluating Excess Returns
- [x] Task B.1: Implement Backtesting Protocol Setups
- [x] Task B.2: Implement Rademacher Complexity and RAS Bounds
- [x] Task B.3: Update `__init__.py` files
- [x] Create example script `examples/chapter_8_evaluating_excess_returns_example.py`
- [x] Create unit tests in `tests/test_backtesting.py`

### Chapter 9: Portfolio Management: The Basics
- [ ] Task B.1: Implement MVO Solvers
- [ ] Task B.2: Implement Factor-Mimicking Portfolio (FMP) Calculation
- [ ] Task B.3: Implement Logic for Adding a New Factor (Procedure 9.1)
- [ ] Task B.4: Implement Factor-Neutral MVO
- [ ] Task B.5: Implement Fundamental Law Calculation
- [x] Task B.6: Implement Matrix Utility Lemmas
- [ ] Task B.7: Update `__init__.py` files.
- [ ] Create example script `examples/chapter_9_mvo_basics_example.py`
- [ ] Create unit tests in `tests/test_portfolio_optimization.py`, `tests/test_factor_models_fmp.py`, `tests/test_utils_linalg.py`

### Chapter 10: Beyond Simple Mean Variance
- [ ] Task B.1: Implement Constrained MVO Solvers using `cvxpy`
- [ ] Task B.2: Implement MVO with L2 Norm Penalty on Weights
- [ ] Task B.3: Implement Sharpe Ratio Efficiency (SRE) Calculation
- [ ] Task B.4: Update `__init__.py` files
- [ ] Create example script `examples/chapter_10_beyond_mvo_example.py`
- [ ] Create unit tests in `tests/test_portfolio_optimization_adv.py`

### Chapter 11: Market Impact-Aware Portfolio Management
- [ ] Task B.1: Implement Market Impact Cost Functions
- [ ] Task B.2: Implement Finite-Horizon Execution Optimizer (Simplified)
- [ ] Task B.3: Implement Infinite-Horizon Execution Optimizer (Procedure 11.1)
- [ ] Task B.4: Update `__init__.py` files.
- [ ] Create example script `examples/chapter_11_market_impact_example.py`
- [ ] Create unit tests in `tests/test_market_impact.py`, `tests/test_execution_optimizers.py`

### Chapter 12: Hedging
- [ ] Task B.1: Implement Simple Two-Asset Hedging Logic
- [ ] Task B.2: Implement Factor Hedging Logic (Conceptual MVO Setup)
- [ ] Task B.3: Implement Hedging with Beta Estimation Error (Shrinkage)
- [ ] Task B.4: Implement Optimal Tracking Portfolio for Time Series
- [ ] Task B.5: Update `__init__.py` files.
- [ ] Create example script `examples/chapter_12_hedging_example.py`
- [ ] Create unit tests in `tests/test_hedging.py`

### Chapter 13: Dynamic Risk Allocation
- [ ] Task B.1: Implement Kelly Criterion Calculations
- [ ] Task B.2: Implement Grossman-Zhou Drawdown Control Policy
- [ ] Task B.3: Update `__init__.py` files.
- [ ] Create example script `examples/chapter_13_dynamic_risk_alloc_example.py`
- [ ] Create unit tests in `tests/test_risk_allocation.py`

### Chapter 14: Ex-Post Performance Attribution
- [ ] Task B.1: Implement Basic PnL Decomposition
- [ ] Task B.2: Implement Attribution Error Estimation
- [ ] Task B.3: Implement Maximal Performance Attribution (Procedure 14.1 & 14.2)
- [ ] Task B.4: Implement Selection vs. Sizing Attribution
- [ ] Task B.5: Update `__init__.py` files.
- [ ] Create example script `examples/chapter_14_performance_attribution_example.py`
- [ ] Create unit tests in `tests/test_performance_attribution.py`

## Phase 3: Integration, Advanced Topics & Refinements
- [ ] **Cross-Chapter Integration & Workflow Examples:**
- [ ] **Advanced Feature Implementation (Iterative Backlog):**
- [ ] **API Review and Refinement:**
- [ ] **Documentation Generation:**
- [x] **Packaging and Distribution (Poetry):**
- [ ] **Performance Optimization (If Necessary):**
- [ ] **Comprehensive Testing:**
