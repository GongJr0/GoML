package OLS

import (
	"GoML/Ensemble"
	"GoML/metrics"
	"fmt"
	"math"
	"slices"

	"gonum.org/v1/gonum/mat"
)

func addInterceptColumn(X [][]float64) [][]float64 {
	n := len(X)
	xNew := make([][]float64, n)
	for i := range X {
		xNew[i] = make([]float64, len(X[i])+1)
		xNew[i][0] = 1.0 // Intercept term
		copy(xNew[i][1:], X[i])
	}
	return xNew
}

type OLS struct {
	X         [][]float64 `json:"X,omitempty"`     // nd-array[float64]
	Y         []float64   `json:"y,omitempty"`     //1d-array[float64]
	Coefs     []float64   `json:"coefs,omitempty"` // 1d-array[float64]
	Intercept float64     `json:"intercept,omitempty"`

	Metrics metrics.Metrics
}

func NewOLS(X [][]float64, Y []float64) Ensemble.Estimator {
	if len(X) == 0 || len(Y) == 0 {
		panic("X and Y cannot be empty")
	}
	if len(X) != len(Y) {
		panic("X and Y must have the same number of rows")
	}
	for i := range X {
		if len(X[i]) == 0 {
			panic("X cannot have empty rows")
		}
	}

	X = addInterceptColumn(X)
	for i, val := range Y {
		if val == 0.0 {
			Y[i] = 1e-8 // Avoid Div0
		}
	}

	preAllocX := make([][]float64, len(X))
	for i := range X {
		preAllocX[i] = make([]float64, len(X[i]))
		copy(preAllocX[i], X[i])
	}

	preAllocY := make([]float64, len(Y))
	copy(preAllocY, Y)

	for i := range Y {
		if math.IsNaN(Y[i]) || math.IsInf(Y[i], 0) {
			panic(fmt.Sprintf("Y contains NaN or Inf at index %d", i))
		}
	}

	nCoefs := len(X)
	coefs := make([]float64, nCoefs)
	for i := range coefs {
		coefs[i] = 0.0
	}

	ols := &OLS{
		X:     preAllocX,
		Y:     preAllocY,
		Coefs: coefs,
	}
	return ols
}

func (ols *OLS) Fit() {
	nRows, nCols := len(ols.X), len(ols.X[0])

	xFlattened := make([]float64, 0, nRows*nCols)
	for _, row := range ols.X {
		xFlattened = append(xFlattened, row...)
	}
	xMatrix := mat.NewDense(nRows, nCols, xFlattened)
	yVector := mat.NewVecDense(nRows, ols.Y)

	var svd mat.SVD
	ok := svd.Factorize(xMatrix, mat.SVDThin)
	if !ok {
		panic("SVD Factorization Failed")
	}

	singularValues := svd.Values(nil)
	eps := 1e-8 // epsilon defined to 1e-8 at y[i]==0.0 check
	rank := 0
	for _, s := range singularValues {
		if s > eps {
			rank++
		}
	}

	var beta mat.Dense
	svd.SolveTo(&beta, yVector, rank)

	raw := beta.RawMatrix().Data
	ols.Intercept = raw[0]
	ols.Coefs = raw[1:]

	// Metrics
	preds := make([]float64, nRows)
	for i := 0; i < nRows; i++ {
		preds[i] = ols.Predict(ols.X[i][1:])
	}

	ols.Metrics = metrics.Evaluate(ols.Y, preds)
}

func (ols *OLS) Predict(x []float64) float64 {
	if len(x) != len(ols.Coefs) {
		panic("Input feature length does not match number of coefficients")
	}

	coefs := make([]float64, len(ols.Coefs)+1)
	coefs[0] = ols.Intercept
	copy(coefs[1:], ols.Coefs)

	x = slices.Concat([]float64{1.0}, x)

	pred := 0.0
	for i, coef := range coefs {
		pred += coef * x[i]
	}
	return pred
}

func PredictorFromFitted(ols *OLS, x []float64) func([]float64) float64 {
	if ols == nil {
		panic("OLS model is nil")
	}
	if len(ols.Coefs) == 0 {
		panic("OLS model is not fitted yet")
	}
	if len(x) != len(ols.Coefs) {
		panic("Input feature length does not match number of coefficients")
	}

	intercept := ols.Intercept
	coefs := make([]float64, len(ols.Coefs)+1)
	copy(coefs[1:], ols.Coefs)
	coefs[0] = intercept

	return func(x []float64) float64 {
		if len(x) != len(coefs) {
			panic("Input feature length does not match number of coefficients")
		}
		x = slices.Concat([]float64{1.0}, x)

		pred := 0.0
		for i, coef := range coefs {
			pred += coef * x[i]
		}
		return pred
	}
}

func (ols *OLS) GetMetrics() metrics.Metrics {
	return ols.Metrics
}
