package Ensemble

type Estimator interface {
	Fit()
	Predict([]float64) float64
}

type Sample struct {
	X          [][]float64
	Y          []float64
	OOBIndices map[int]bool
}
