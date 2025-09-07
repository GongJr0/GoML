package httpServer

import (
	"GoML/DecTree"
	"GoML/Ensemble"
	"GoML/LinReg"
	"GoML/OLS"
	"GoML/metrics"
	"encoding/json"
	"errors"
	"net/http"
)

func ensembleFactoryConstructor(baseModel string, baseEstimatorParams map[string]interface{}) (func(x [][]float64, y []float64) Ensemble.Estimator, error) {
	var baseEstimatorFactory func(x [][]float64, y []float64) Ensemble.Estimator
	switch baseModel {
	case "linreg":
		baseEstimatorFactory = LinReg.NewLinReg
	case "ols":
		baseEstimatorFactory = OLS.NewOLS
	case "dectree":
		baseEstimatorFactory = func(x [][]float64, y []float64) Ensemble.Estimator {
			randSeed := int64(baseEstimatorParams["random_seed"].(float64))
			maxFeatures := int(baseEstimatorParams["max_features"].(float64))
			return DecTree.NewDecTree(x, y,
				int(baseEstimatorParams["max_depth"].(float64)),
				int(baseEstimatorParams["min_samples_split"].(float64)),
				int(baseEstimatorParams["min_samples_leaf"].(float64)),
				&randSeed,
				&maxFeatures)
		}
	default:
		return nil, errors.New("unsupported base estimator")
	}
	return baseEstimatorFactory, nil
}

type AbstractPostBody struct {
	X [][]float64 `json:"X"`
	Y []float64   `json:"Y"`
}

type DecTreePostBody struct {
	AbstractPostBody
	MaxDepth        int   `json:"max_depth"`
	MinSamplesSplit int   `json:"min_samples_split"`
	MinSamplesLeaf  int   `json:"min_samples_leaf"`
	MaxFeatures     int   `json:"max_features"`
	RandomSeed      int64 `json:"random_seed"`
}

type EnsemblePostBody struct {
	AbstractPostBody
	BaseEstimator       string                 `json:"base_estimator"`
	BaseEstimatorParams map[string]interface{} `json:"base_estimator_params"`
	NEstimators         int                    `json:"n_estimators"`
	RandomSeed          int64                  `json:"random_seed,omitempty"`   // for bagged
	LearningRate        float64                `json:"learning_rate,omitempty"` // for boosted
}

// Documentation as JSON response for each endpoint

var metricsDescription = map[string]string{
	"r2":   "R-squared, the coefficient of determination.",
	"mse":  "Mean Squared Error.",
	"rmse": "Root Mean Squared Error.",
	"mae":  "Mean Absolute Error.",
	"mape": "Mean Absolute Percentage Error.",
}

var defaultBody = map[string]interface{}{
	"X": "[[feature1, feature2, ...], [feature1, feature2, ...], ...]",
	"Y": "[target]",
}

var ensembleMethods = []string{"bagged", "boosted"}

var models = []string{"linreg", "ols", "dectree"}

var linRegDocs = map[string]interface{}{
	"description":      "Basic linear regression with no configurable arguments.",
	"params":           []string{},
	"ensemble_support": true,
	"ensemble_methods": ensembleMethods,
	"request_format": map[string]interface{}{
		"type": "POST",
		"body": defaultBody,
		"response": map[string]interface{}{
			"coefficients": "[coef1, coef2, ...]",
			"intercept":    "intercept",
			"fit_metrics":  metricsDescription,
		},
	},
}

var olsDocs = map[string]interface{}{
	"description":      "Ordinary Least Squares regression with no configurable arguments.",
	"params":           []string{},
	"ensemble_support": true,
	"ensemble_methods": ensembleMethods,
	"request_format": map[string]interface{}{
		"type": "POST",
		"body": defaultBody,
	},
	"response": map[string]interface{}{
		"coefficients": "[coef1, coef2, ...]",
		"intercept":    "intercept",
		"fit_metrics":  metricsDescription,
	},
}

var decTreeDocs = map[string]interface{}{
	"description": "Decision Tree regression with configurable arguments.",
	"params": map[string][]string{
		"max_depth":         {"int", "Maximum depth of the tree. Default is 5."},
		"min_samples_split": {"int", "Minimum number of samples required to split an internal node. Default is 2."},
		"min_samples_leaf":  {"int", "Minimum number of samples required to be at a leaf node. Default is 1."},
		"max_features":      {"int", "Number of features to consider when looking for the best split. Default is all features."},
		"random_seed":       {"int", "Random seed for reproducibility. Default is current unix time in nanoseconds."},
	},
	"ensemble_support": true,
	"ensemble_methods": ensembleMethods,
	"request_format": map[string]interface{}{
		"type": "POST",
		"body": map[string]string{
			"X":                 "[[feature1, feature2, ...], [feature1, feature2, ...], ...]",
			"Y":                 "[target]",
			"max_depth":         "int",
			"min_samples_split": "int",
			"min_samples_leaf":  "int",
			"max_features":      "int",
			"random_seed":       "int",
		},
		"response": map[string]interface{}{
			"tree_structure":     "{...}",
			"feature_importance": "[imp1, imp2, ...]",
			"fit_metrics":        metricsDescription,
		},
	},
}

var baggedDocs = map[string]interface{}{
	"description": "Bagging ensemble method. Combines the predictions of multiple base estimators trained on random subsets of the data.",
	"params": map[string][]string{
		"n_estimators": {"int", "The number of base estimators in the ensemble. Default is 10."},
		"random_seed":  {"int", "Random seed for reproducibility. Default is current unix time in nanoseconds."},
	},
	"supported_base_estimators": models,
	"request_format": map[string]interface{}{
		"type": "POST",
		"body": map[string]string{
			"X":                     "[[feature1, feature2, ...], [feature1, feature2, ...], ...]",
			"Y":                     "[target]",
			"base_estimator":        "'linreg' | 'ols' | 'dectree'",
			"base_estimator_params": "{...} // {} if no params",
			"n_estimators":          "int",
			"random_seed":           "int",
		},
		"response": map[string]interface{}{
			"base_estimator_fit_metrics": "[{...}, {...}, ...]",
			"fit_metrics":                metricsDescription,
		},
	},
}

var boostedDocs = map[string]interface{}{
	"description": "Boosting ensemble method. Combines the predictions of multiple base estimators trained sequentially, where each estimator tries to correct the errors of the previous one.",
	"params": map[string][]string{
		"n_estimators":  {"int", "The number of base estimators in the ensemble. Default is 10."},
		"learning_rate": {"float", "Learning rate shrinks the contribution of each base estimator. Default is 0.1."},
	},
	"supported_base_estimators": models,
	"request_format": map[string]interface{}{
		"type": "POST",
		"body": map[string]string{
			"X":                     "[[feature1, feature2, ...], [feature1, feature2, ...], ...]",
			"Y":                     "[target]",
			"base_estimator":        "'linreg' | 'ols' | 'dectree'",
			"base_estimator_params": "{...} // {} if no params",
			"n_estimators":          "int",
			"learning_rate":         "float",
		},
		"response": map[string]interface{}{
			"base_estimator_fit_response": "[{...}, {...}, ...]",
			"fit_metrics":                 metricsDescription,
		},
	},
}

var endpointUsage = map[string]interface{}{
	"estimators": map[string]interface{}{
		"/linreg":  linRegDocs,
		"/ols":     olsDocs,
		"/dectree": decTreeDocs,
	},
	"ensembles": map[string]interface{}{
		"/bagged":  baggedDocs,
		"/boosted": boostedDocs,
	},
}

// GET handlers for each endpoint to return the documentation as JSON

func ModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Only GET is allowed on the /models endpoint", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(endpointUsage)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	return
}

func EstimatorsHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(endpointUsage["estimators"])
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	return
}

func LinRegGetHandler(w http.ResponseWriter, r *http.Request) (err error) {
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(linRegDocs)
	return
}

func OLSGetHandler(w http.ResponseWriter, r *http.Request) (err error) {
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(olsDocs)
	return
}

func DecTreeGetHandler(w http.ResponseWriter, r *http.Request) (err error) {
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(decTreeDocs)
	return
}

func EnsemblesHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(endpointUsage["ensembles"])
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	return
}

func BaggedGetHandler(w http.ResponseWriter, r *http.Request) (err error) {
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(baggedDocs)
	return
}

func BoostedGetHandler(w http.ResponseWriter, r *http.Request) (err error) {
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(boostedDocs)
	return
}

// POST handlers for each endpoint to handle model training and prediction

func LinRegPostHandler(w http.ResponseWriter, r *http.Request) (err error) {
	var modelParams AbstractPostBody
	err = json.NewDecoder(r.Body).Decode(&modelParams)
	if err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}
	X := modelParams.X
	Y := modelParams.Y

	model := LinReg.NewLinReg(X, Y).(*LinReg.LinReg)
	model.Fit()

	resp := map[string]interface{}{
		"coefficients": model.Coefs,
		"fit_metrics":  model.GetMetrics(),
	}
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(resp)
	return
}

func OLSPostHandler(w http.ResponseWriter, r *http.Request) (err error) {
	var modelParams AbstractPostBody
	err = json.NewDecoder(r.Body).Decode(&modelParams)
	if err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}
	X := modelParams.X
	Y := modelParams.Y

	model := OLS.NewOLS(X, Y).(*OLS.OLS)
	model.Fit()

	resp := map[string]interface{}{
		"coefficients": model.Coefs,
		"intercept":    model.Intercept,
		"fit_metrics":  model.GetMetrics(),
	}
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(resp)
	return
}

func DecTreePostHandler(w http.ResponseWriter, r *http.Request) (err error) {
	var modelParams DecTreePostBody
	err = json.NewDecoder(r.Body).Decode(&modelParams)
	if err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}
	X := modelParams.X
	Y := modelParams.Y

	maxDepth := modelParams.MaxDepth
	minSamplesSplit := modelParams.MinSamplesSplit
	minSamplesLeaf := modelParams.MinSamplesLeaf
	maxFeatures := modelParams.MaxFeatures
	randomSeed := modelParams.RandomSeed

	model := DecTree.NewDecTree(X, Y, maxDepth, minSamplesSplit, minSamplesLeaf, &randomSeed, &maxFeatures).(*DecTree.DecTree)
	model.Fit()

	resp := map[string]interface{}{
		"tree_structure":     model.GetTreeString(),
		"feature_importance": model.GetFeatureImportance(),
		"fit_metrics":        model.GetMetrics(),
	}
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(resp)
	return
}

func BaggedPostHandler(w http.ResponseWriter, r *http.Request) (err error) {
	var modelParams EnsemblePostBody
	err = json.NewDecoder(r.Body).Decode(&modelParams)
	if err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}
	X := modelParams.X
	Y := modelParams.Y
	baseEstimatorName := modelParams.BaseEstimator
	baseEstimatorParams := modelParams.BaseEstimatorParams
	nEstimators := modelParams.NEstimators
	randomSeed := modelParams.RandomSeed

	baseEstimatorFactory, err := ensembleFactoryConstructor(baseEstimatorName, baseEstimatorParams)
	if err != nil {
		http.Error(w, "Unsupported base estimator", http.StatusBadRequest)
		return
	}
	ensemble := Ensemble.NewBagged(baseEstimatorFactory, nEstimators, X, Y, &randomSeed).(*Ensemble.Bagged)
	ensemble.Fit()

	estimatorFits := make([]metrics.Metrics, nEstimators)
	for i, est := range ensemble.Estimators {
		estimatorFits[i] = est.GetMetrics()
	}

	resp := map[string]interface{}{
		"base_estimator_fit_metrics": estimatorFits,
		"fit_metrics":                ensemble.GetMetrics(),
	}
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(resp)
	return
}

func BoostedPostHandler(w http.ResponseWriter, r *http.Request) (err error) {
	var modelParams EnsemblePostBody
	err = json.NewDecoder(r.Body).Decode(&modelParams)
	if err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}
	X := modelParams.X
	Y := modelParams.Y
	baseEstimatorName := modelParams.BaseEstimator
	baseEstimatorParams := modelParams.BaseEstimatorParams
	nEstimators := modelParams.NEstimators
	learningRate := modelParams.LearningRate

	baseEstimatorFactory, err := ensembleFactoryConstructor(baseEstimatorName, baseEstimatorParams)
	if err != nil {
		http.Error(w, "Unsupported base estimator", http.StatusBadRequest)
		return
	}
	ensemble := Ensemble.NewBoosted(baseEstimatorFactory, nEstimators, X, Y, learningRate).(*Ensemble.Boosted)
	ensemble.Fit()

	estimatorFits := make([]metrics.Metrics, nEstimators)
	for i, est := range ensemble.Estimators {
		estimatorFits[i] = est.GetMetrics()
	}

	resp := map[string]interface{}{
		"base_estimator_fit_response": estimatorFits,
		"fit_metrics":                 ensemble.GetMetrics(),
	}
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(resp)
	return
}
