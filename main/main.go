package main

import (
	"GoML/OLS"
	"encoding/json"
	"fmt"
	"strings"
)

func main() {
	dummyX := [][]float64{
		{1, 2},
		{2, 3},
		{3, 4},
		{4, 5},
	}
	dummyY := []float64{2, 3, 4, 5}

	model := OLS.NewOLS(dummyX, dummyY)
	model.Fit()

	fmt.Println("Intercept:", model.Intercept)
	fmt.Println("Coefficients:", model.Coefs)
	metricsJSON, _ := json.Marshal(model.Metrics)
	metricsFormatted := strings.Join(strings.Split(strings.Replace(string(metricsJSON), "\"", "", -1), ","), "\n")
	fmt.Println("Metrics: ", metricsFormatted)

	// Make a prediction
	testData := []float64{5, 6}
	prediction := model.Predict(testData)
	fmt.Printf("Prediction for input %v: %v\n", testData, prediction)
}
