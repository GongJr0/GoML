package LinReg

import (
	"GoML/Ensemble"
	"GoML/metrics"

	"gonum.org/v1/gonum/mat"
)

type LinReg struct {
	X     [][]float64
	Y     []float64
	Coefs []float64

	Metrics metrics.Metrics
}

func NewLinReg(X [][]float64, Y []float64) Ensemble.Estimator {
	preAllocX := make([][]float64, 0, len(X))
	for i := range X {
		x := make([]float64, len(X[i]))
		copy(x, X[i])
		preAllocX = append(preAllocX, x)
	}

	preAllocY := make([]float64, 0, len(Y))
	for _, val := range Y {
		if val == 0.0 {
			val = 1e-8 // Avoid Div0
		}
		preAllocY = append(preAllocY, val)
	}
	preAllocCoefs := make([]float64, len(X[0]))

	return &LinReg{
		X:     preAllocX,
		Y:     preAllocY,
		Coefs: preAllocCoefs,
	}

}

func (lr *LinReg) Fit() {
	var xFlattened []float64
	for _, row := range lr.X {
		xFlattened = append(xFlattened, row...)
	}

	xMatrix := mat.NewDense(len(lr.X), len(lr.X[0]), xFlattened)
	yMatrix := mat.NewVecDense(len(lr.Y), lr.Y)

	var svd mat.SVD
	ok := svd.Factorize(xMatrix, mat.SVDThin)
	if !ok {
		panic("SVD factorization failed")
	}
	svdValues := svd.Values(nil)
	eps := 1e-8
	rank := 0
	for _, val := range svdValues {
		if val > eps {
			rank++
		}
	}

	var W mat.Dense
	svd.SolveTo(&W, yMatrix, rank)

	raw := W.RawMatrix().Data
	copy(lr.Coefs, raw)

	preds := make([]float64, len(lr.Y))
	for i := range lr.Y {
		preds[i] = lr.Predict(lr.X[i])
	}

	lr.Metrics = metrics.Evaluate(lr.Y, preds)
}

func (lr *LinReg) Predict(x []float64) float64 {
	pred := 0.0
	for i, val := range x {
		pred += lr.Coefs[i] * val
	}
	return pred
}

func PredictorFromFitted(lr *LinReg, x []float64) func(x []float64) float64 {
	if lr == nil {
		panic("OLS model is nil")
	}
	if len(lr.Coefs) == 0 {
		panic("OLS model is not fitted yet")
	}
	if len(x) != len(lr.Coefs) {
		panic("Input feature length does not match number of coefficients")
	}

	var coefs []float64
	copy(coefs, lr.Coefs)

	return func(x []float64) float64 {
		pred := 0.0
		for i, val := range x {
			pred += coefs[i] * val
		}
		return pred
	}
}

func (lr *LinReg) GetMetrics() metrics.Metrics {
	return lr.Metrics
}
