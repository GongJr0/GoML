package Ensemble

import (
	"GoML/metrics"
	"math"
)

type Boosted struct {
	X [][]float64
	Y []float64

	Estimators   []Estimator
	Factory      func(x [][]float64, y []float64) Estimator
	LearningRate float64

	Metrics metrics.Metrics
}

func NewBoosted(estimatorFactory func(x [][]float64, y []float64) Estimator, nEstimators int, x [][]float64, y []float64, learningRate float64) Estimator {
	estimators := make([]Estimator, nEstimators)
	estimators[0] = estimatorFactory(x, y)

	return &Boosted{
		X:            x,
		Y:            y,
		Estimators:   estimators,
		Factory:      estimatorFactory,
		LearningRate: learningRate,
	}
}

func NewDefaultBoosted(estimatorFactory func(x [][]float64, y []float64) Estimator, nEstimators int, x [][]float64, y []float64) Estimator {
	return NewBoosted(estimatorFactory, nEstimators, x, y, 0.1).(*Boosted)
}
func (b *Boosted) Fit() {
	nEstimators := len(b.Estimators)

	prevSSR := 0.0
	for i := 0; i < nEstimators; i++ {
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

			SSR := 0.0
			for _, r := range resid {
				SSR += r * r
			}
			if math.Abs((SSR-prevSSR)/(prevSSR+1e-6)) < 5e-4 {
				estimatorArr := make([]Estimator, i)
				copy(estimatorArr, b.Estimators[:i])
				b.Estimators = estimatorArr
				b.Metrics = metrics.Evaluate(b.Y, preds)
				return
			}

			b.Estimators[i] = b.Factory(b.X, resid)
		}

		b.Estimators[i].Fit()
	}
	preds := make([]float64, len(b.Y))
	for i, row := range b.X {
		preds[i] = b.Predict(row)
	}
	b.Metrics = metrics.Evaluate(b.Y, preds)
}

func (b *Boosted) Predict(x []float64) float64 {
	pred := 0.0
	for _, est := range b.Estimators {
		pred += est.Predict(x) * b.LearningRate
	}
	return pred
}

func (b *Boosted) GetMetrics() metrics.Metrics {
	return b.Metrics
}
