package demo

import (
	"GoML/DecTree"
	"GoML/Ensemble"
	"GoML/LinReg"
	"GoML/OLS"

	"encoding/json"
	"fmt"
	"strings"
)

var Models = map[string]func(x [][]float64, y []float64) Ensemble.Estimator{
	"linreg":  LinReg.NewLinReg,
	"ols":     OLS.NewOLS,
	"dectree": DecTree.NewDefaultDecTree,
}

var EnsembleType = map[string]func(estimatorFactory func(x [][]float64, y []float64) Ensemble.Estimator, nEstimators int, x [][]float64, y []float64) Ensemble.Estimator{
	"bagged":  Ensemble.NewDefaultBagged,
	"boosted": Ensemble.NewBoosted,
}

func Run(dummyX [][]float64, dummyY []float64, modelName string, isEnsemble bool, ensembleMethod string, nEstimators int) {
	var model Ensemble.Estimator

	if isEnsemble {
		factory := Models[strings.ToLower(modelName)]
		ensembleFactory := EnsembleType[strings.ToLower(ensembleMethod)](factory, nEstimators, dummyX, dummyY)
		model = ensembleFactory
		model.Fit()
	} else {
		model = Models[strings.ToLower(modelName)](dummyX, dummyY)
		model.Fit()
	}

	//fmt.Println("Coefficients: ", model.Coef)
	metricsJSON, _ := json.Marshal(model.GetMetrics())
	metricsFormatted := strings.Join(strings.Split(strings.Replace(string(metricsJSON), "\"", "", -1), ","), "\n")
	fmt.Println("Metrics: ", metricsFormatted)

	// Make a prediction
	testData := dummyX[len(dummyX)-1]
	prediction := model.Predict(testData)
	fmt.Printf("Prediction for input %v: %v\n", testData, prediction)
	return
}
