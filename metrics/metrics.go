package metrics

import (
	"math"

	"gonum.org/v1/gonum/stat"
)

type Metrics struct {
	R2   float64 `json:"R2"`
	MSE  float64 `json:"MSE"`
	RMSE float64 `json:"RMSE"`
	MAE  float64 `json:"MAE"`
	MAPE float64 `json:"MAPE"`
}

func Evaluate(yTrue, yPred []float64) Metrics {
	var R2, MSE, RMSE, MAE, MAPE float64

	SSR := 0.0
	SST := 0.0
	AE := 0.0
	APE := 0.0

	nRows := len(yTrue)
	yMean := stat.Mean(yTrue, nil)

	for i := 0; i < nRows; i++ {
		SSR += math.Pow(yPred[i]-yTrue[i], 2)
		SST += math.Pow(yTrue[i]-yMean, 2)
		AE += math.Abs(yPred[i] - yTrue[i])
		APE += math.Abs((yPred[i] - yTrue[i]) / yTrue[i])
	}

	//R2
	R2 = 1 - (SSR / SST)

	//MSE
	MSE = SSR / float64(nRows)
	RMSE = math.Sqrt(MSE)

	MAE = AE / float64(nRows)
	MAPE = APE / float64(nRows)

	return Metrics{
		R2:   R2,
		MSE:  MSE,
		RMSE: RMSE,
		MAE:  MAE,
		MAPE: MAPE,
	}
}
