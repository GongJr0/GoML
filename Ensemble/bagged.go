package Ensemble

import (
	"GoML/metrics"
	"math/rand"
	"slices"
	"time"

	"gonum.org/v1/gonum/stat"
)

type Bagged struct {
	//Ensemble Components
	Estimators []Estimator
	Bags       []Sample
	weights    []float64

	// Raw Data
	X [][]float64
	Y []float64

	//Metrics
	FitMetrics metrics.Metrics
	OOBMetrics metrics.Metrics

	// Random State
	RandSeed *int64
	rng      *rand.Rand
}

func NewBagged(estimatorFactory func(x [][]float64, y []float64) Estimator, nEstimators int, x [][]float64, y []float64, randSeed *int64) Estimator {
	if randSeed == nil {
		tNow := time.Now().UnixNano()
		randSeed = &tNow
	}

	b := &Bagged{
		X:        x,
		Y:        y,
		RandSeed: randSeed,
		rng:      rand.New(rand.NewSource(*randSeed)),
	}
	b.setBags(nEstimators)
	estimators := make([]Estimator, nEstimators)
	for i := 0; i < nEstimators; i++ {
		estimators[i] = estimatorFactory(b.Bags[i].X, b.Bags[i].Y)
	}
	b.Estimators = estimators

	return b
}

func NewDefaultBagged(estimatorFactory func(x [][]float64, y []float64) Estimator, nEstimators int, x [][]float64, y []float64) Estimator {
	return NewBagged(estimatorFactory, nEstimators, x, y, nil)
}

func (b *Bagged) bootstrapSample() Sample {
	nRows := len(b.Y)

	feature := make([][]float64, nRows)
	label := make([]float64, nRows)
	indices := make([]int, 0, nRows)
	for i := 0; i < nRows; i++ {
		idx := b.rng.Intn(nRows)
		feature[i] = b.X[idx]
		label[i] = b.Y[idx]
		indices = append(indices, idx)
	}

	oobIndices := make(map[int]bool)
	for i := 0; i < nRows; i++ {
		oobIndices[i] = !slices.Contains(indices, i)
	}

	return Sample{X: feature, Y: label, OOBIndices: oobIndices}
}

func (b *Bagged) setBags(nEstimators int) {
	bags := make([]Sample, nEstimators)
	for i := 0; i < nEstimators; i++ {
		bags[i] = b.bootstrapSample()
	}
	b.Bags = bags
}

func (b *Bagged) GetOOB() []Sample {
	samples := make([]Sample, len(b.Estimators))
	for i := 0; i < len(b.Estimators); i++ {
		oobX := make([][]float64, 0)
		oobY := make([]float64, 0)

		for idx, isOOB := range b.Bags[i].OOBIndices {
			if isOOB {
				oobX = append(oobX, b.X[idx])
				oobY = append(oobY, b.Y[idx])
			}
			s := Sample{
				X:          oobX,
				Y:          oobY,
				OOBIndices: b.Bags[i].OOBIndices,
			}
			samples[i] = s
		}
	}
	return samples
}

func (b *Bagged) Fit() {
	oobEval := make([]float64, len(b.Estimators))

	for _, estimator := range b.Estimators {
		estimator.Fit()
	}
	oob := b.GetOOB()
	evalSetOOB := make([]metrics.Metrics, len(b.Estimators))
	for i, sample := range oob {
		preds := make([]float64, len(sample.Y))
		for row, x := range sample.X {
			preds[row] = b.Estimators[i].Predict(x)
		}
		evalSetOOB[i] = metrics.Evaluate(sample.Y, preds)
		oobEval[i] = 1 / (evalSetOOB[i].RMSE + 1e-8)
	}
	weights := make([]float64, len(oobEval))
	sumMetric := 0.0
	for _, metric := range oobEval {
		sumMetric += metric
	}
	for estimator, metric := range oobEval {
		weights[estimator] = metric / sumMetric
	}
	b.weights = weights

	metricsOOB := metrics.Metrics{}
	for i, metric := range evalSetOOB {
		metricsOOB.R2 += metric.R2 * weights[i]
		metricsOOB.RMSE += metric.RMSE * weights[i]
		metricsOOB.MSE += metric.MSE * weights[i]
		metricsOOB.MAE += metric.MAE * weights[i]
		metricsOOB.MAPE += metric.MAPE * weights[i]
	}

	predsFit := make([]float64, len(b.Y))
	for i := range b.Y {
		predsFit[i] = b.Predict(b.X[i])
	}
	metricsFit := metrics.Evaluate(b.Y, predsFit)

	b.FitMetrics = metricsFit
	b.OOBMetrics = metricsOOB

}

func (b *Bagged) Predict(x []float64) float64 {
	preds := make([]float64, len(b.Estimators))
	for i, estimator := range b.Estimators {
		preds[i] = estimator.Predict(x)
	}
	return stat.Mean(preds, b.weights)
}

func (b *Bagged) GetMetrics() metrics.Metrics {
	return b.FitMetrics
}
