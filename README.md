# GoML: A simple, lightweight ML package with RESTful HTML integration

GoML, primarily a learning project, is a collection of simple ML models and interfacing tools. 
As of now, the package explicitly focuses on supervised regression algorithms (the scope of functionality will expand as development goes on) with 3 available models and ensemble capabilities.
A CLI and a RESTful API is implemented to provide interfacing in local and remote access environments.


## Available Models

### Linear Regression
Linear Regression is a staple of entry-level ML algorithms with a very understandable operating logic and clear assumptions/limitations. 
Given a suitable regression task, it can be as powerful as complex models for a fraction of the resources. 
However, linear regression is a generally limited model and is rarely the correct choice for a real-life application where assumption of strict linearity are often shaky.

To make the most out of linear regression, focusing on feature selection is essential. 
This model is not known for its robustness, reducing multicollinearily, isolating linear correlations, and feature re-scaling are often very beneficial for model performance.

GoML's implementation of linear regression uses $SSR$ minimization (This makes linear regression a subset of the OLS Regressor shown below) and does not contain an intercept term (often denoted: $\beta_0$); using a pure sumproduct of weighted features:

$$y=\sum_{i=1}^n \beta_i X_i$$

The model is defined as:
```go
type LinReg struct {
X [][]float64
Y []float64
Coefs []float64
metrics metrics.Metrics
}
```
Where `Coefs` stores the coefficients determined at fit time and `metrics.Metrics` stores error metrics ($R^2$, $MSE$, $RMSE$, $MAE$, and $MAPE$). 
GoML's linear regression can fit, generate predictions, and evaluate errors of a dataset with dimensions (30, 13) with a response time of 3ms over `localhost`. (Tested with a slice of the publicly available [boston housing dataset](https://lib.stat.cmu.edu/datasets/boston))

### OLS Regression
Ordinary Least Squares (OLS) is very closely related to Linear Regression. In GoML, OLS is differentiated from linear regression through the inclusion of an intercept parameter. 
It shares Linear Regression's limitations, but can fit better with features that are separated from the target by an additive constant in scale. 
Through the added intercept coefficient, OLS introduces an extra column to all matrix operations compared to linear regression, and the two is separated simply to keep linear regression as lean as possible to preserve minimum latency where possible.
The fit equation of OLS can be written as:

$$y=\beta_0+\sum_{i=1}^n\beta_iX_i$$
$$\text{or}$$
$$y=\sum_{i=0}^n \beta_i X_i \\ \\ \text{where } X_0=1 \text{   (GoML's implementation)}$$

OLS shares a similar model definition to linear regression but includes the added intercept parameter:
```go
type OLS struct {
X [][]float64
Y []float64
Coefs []float64
Intercept float64
metrics metrics.Metrics
}
```

### Decision Tree Regression

The decision tree algorithm is a binary tree with probabilistic splits based on given feature values.
A "split" in this context is a binary decision, set to use the most "optimal" feature for the given depth of each node in the tree.
Intuitively, a Decision Tree is similar to a mathematically proven way of playing the "20 questions" game (a game where players ask 20 yes/no questions to guess something) to reach a final verdict.
Traditionally, and in GoML's implementation, each split is made on a single feature, where traversing the tree from top-to-bottom yields the distinct route defined by results of all feature splits.
This unique methodology to regression creates an inherent feature independence and makes the model more robust to scale differences and multicollinearity.
By extension, there's no global target of minimization in a Decision Tree; instead, minimization occurs in splits, targeting feature impurity (how uniform are the target values in a split).
By definition, [Variance Reduction](https://en.wikipedia.org/wiki/Variance_reduction) methods (used for split-wise optimization in regression tasks) decrease impurity by decreasing the average difference between the target values.
In short, optimization is based on a compilation of local minima, rather than a "monolithic" global optimization target.
Decision Trees (or [Random Forests](https://en.wikipedia.org/wiki/Random_forest)) are often the choice of model where feature behavior can be ambiguous or varying over-time due to this robustness.
However, its important to note that this process often computationally inefficient and a model selected to specifically suit the given task can produce similar results in less compute power in most cases.

As there's no clear equation being minimized in the global scope of a Decison Tree, the model is presented in a slightly different fashion in GoML. 
Feature importances (how often a features appears in splits) and the tree structure are often used to quantize the model into a digestible summary. 
GoML also takes this route by exposing a tree structure as a string, and feature importances as percent values in the form $\frac{N_i}{\sum_{j=0}^n N_j}$ where $N_i$ the count of splits on $X_i$ within the whole tree and $\sum_{j=0}^n N_j$ is the total count of splits.
GoML defines the `DecTree` struct as:

```go
type DecTree struct {
	X               [][]float64
	Y               []float64
	Metrics         metrics.Metrics
	root            *Node // Standard tree node struct for binary trees
	MaxDepth        int
	MinSamplesSplit int
	MinSamplesLeaf  int
	MaxFeatures     *int
	RandomSeed      *int64
	rng             *rand.Rand
}
```

## Ensemble Methods
Ensemble estimators are created through combining multiple instances of identical base estimators. GoML implements two primary methods of ensemble generation.

### Bagging
Bagging is an ensemble method that takes advantage of a probabilistic law creating the basis of modern statistics. 
[Law of Large Numbers (LLN)](https://en.wikipedia.org/wiki/Law_of_large_numbers) is a law stating that increasing the number of samples in a set of observations will cause the sample mean to converge to the expected value.
More formally, we can denote an over-simplified $LLN: \\ \\ \lim_{n\to\infty} \overline{X} = \mu$ where $\overline{X}$ is the sample mean and $\mu$ is the population mean.

LLN presents itself in the form of error compensation. Bagging uses multiple, independent models fitted to randomly selected sub-samples of the input data. 
Each model generates a prediction and a weighted average of the individual predictions is calculated, either by equal weights (common default) or by using an error metric ($RMSE$ in GoML's case) to determine which models deserve more weightage.
As a result, multiple models with distinct error profiles are combined additively, resulting in an expected cancellation of errors and a (theoretically) better prediction compared to a single model instance.

GoML's Bagging struct is defined as:
```go
type Bagged struct {
	//Ensemble Components
	Estimators []Estimator
	Bags       []Sample
	weights    []float64  // RMSE based

	// Raw Data
	X [][]float64 
	Y []float64

	//Metrics
	FitMetrics metrics.Metrics // Metrics at fit time 
	OOBMetrics metrics.Metrics // Metrics calculated through data that remained outside the sample for each bag

	// Random State
	RandSeed *int64
	rng      *rand.Rand
}
```

### Boosting

If we define Bagging as a horizontal ensemble (increasing the count of identical models) Boosting is the exact opposite. 
Boosting is built on the idea that one model's shortcomings can be compensated by another.
The Boosting process starts with a single estimator fitted to the data as usual. Of course, the initial estimator produces some error in the form of residuals ($y_{pred}-y_{true}$) which we wish to minimize.
Using the same features, Boosting initializes a new model instance, but trains it on the residuals of the previous model, creating a chain where each model fixes the mistakes of its predecessors.
Typically, this process involves optimization algorithms in multi-dimensional planes such as [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) but GoML takes a much simpler approach sequentially chaining models together.
Boosting is much more fragile and requires tuning of a specific parameter commonly called "Learning Rate".
Learning rate defines the proportion of residuals each model is exposed to.
Mathematically, taking the example learning rate of 10%, define $y_i=(pred_{i-1}-true_{i-1})*0.1$ where $y_i$ is the target for the current model, $pred_{i-1}$ is the predictions of the previous model, and $true_{i-1}$ is the true values the previous model was fitted against.

GoML defines it's Boosting struct as:
```go
type Boosted struct {
  // Ensemble Components
  Estimators   []Estimator
	Factory      func(x [][]float64, y []float64) Estimator
	LearningRate float64

  // Raw Data
	X [][]float64
	Y []float64

  // Metrics
	Metrics metrics.Metrics
}
```

