package Ensemble

import (
	"GoML/metrics"
)

type Boosted struct {
	X [][]float64
	Y []float64

	Estimators []Estimator
	Factory    func(x [][]float64, y []float64) Estimator

	Metrics metrics.Metrics
}

func NewBoosted(estimatorFactory func(x [][]float64, y []float64) Estimator, nLayers int, x [][]float64, y []float64) *Boosted {
	estimators := make([]Estimator, nLayers)
	estimators[0] = estimatorFactory(x, y)

	return &Boosted{
		X:          x,
		Y:          y,
		Estimators: estimators,
		Factory:    estimatorFactory,
	}
}

func (b *Boosted) Fit() {
	nLayers := len(b.Estimators)

	for i := 0; i < nLayers; i++ {
		if i == 0 {
			b.Estimators[i] = b.Factory(b.X, b.Y)
		} else {
			preds := make([]float64, len(b.Y))
			for j, row := range b.X {
				preds[j] = b.Estimators[i-1].Predict(row)
			}

			resid := make([]float64, len(b.Y))
			for j := range b.Y {
				resid[j] = b.Y[j] - preds[j]
			}

			b.Estimators[i] = b.Factory(b.X, resid)
		}

		b.Estimators[i].Fit()
	}
}

func (b *Boosted) Predict(x []float64) float64 {
	pred := 0.0
	for _, est := range b.Estimators {
		pred += est.Predict(x)
	}
	return pred
}
