package demo

import (
	"GoML/LinReg"
	"strings"

	//"GoML/OLS"
	"GoML/Ensemble"
	"encoding/json"
	"fmt"
	//"strings"
)

func Run() {
	dummyX := [][]float64{
		// {m^2, bedRooms, bathrooms, age, garageSpace}
		{850, 2, 1, 30, 1},
		{950, 2, 1, 20, 1},
		{1200, 3, 2, 25, 1},
		{1500, 3, 2, 10, 2},
		{1700, 4, 2, 15, 2},
		{2000, 4, 3, 8, 2},
		{2200, 4, 3, 12, 2},
		{2500, 5, 3, 5, 3},
		{2700, 5, 3, 3, 3},
		{3000, 5, 4, 2, 3},
		{3200, 6, 4, 1, 3},
		{3500, 6, 4, 10, 3},
		{3800, 6, 4, 15, 3},
		{4000, 6, 4, 20, 3},
		{4200, 7, 5, 5, 3},
	}
	dummyY := []float64{145000, 155000, 210000, 260000, 295000,
		345000, 380000, 430000, 460000, 510000,
		545000, 595000, 640000, 670000, 700000}

	factory := LinReg.NewLinReg
	seed := int64(0)
	model := Ensemble.NewBagged(factory, 10, dummyX, dummyY, &seed)
	model.Fit()

	//fmt.Println("Coefficients: ", model.Coef)
	metricsJSON, _ := json.Marshal(model.OOBMetrics)
	metricsFormatted := strings.Join(strings.Split(strings.Replace(string(metricsJSON), "\"", "", -1), ","), "\n")
	fmt.Println("Metrics: ", metricsFormatted)

	// Make a prediction
	testData := []float64{4200, 7, 5, 5, 3}
	prediction := model.Predict(testData)
	fmt.Printf("Prediction for input %v: %v\n", testData, prediction)
}
