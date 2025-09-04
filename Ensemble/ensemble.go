package Ensemble

import "GoML/metrics"

type Estimator interface {
	Fit()
	Predict([]float64) float64
	GetMetrics() metrics.Metrics
}

type Sample struct {
	X          [][]float64
	Y          []float64
	OOBIndices map[int]bool
}
