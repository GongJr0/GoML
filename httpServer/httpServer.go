package httpServer

import (
	"net/http"
)

func AbstractHandler(getHandler, postHandler func(w http.ResponseWriter, r *http.Request) error) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			err := getHandler(w, r)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		case "POST":
			err := postHandler(w, r)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
	}
}

var LinRegHandler = AbstractHandler(LinRegGetHandler, LinRegPostHandler)
var OLSHandler = AbstractHandler(OLSGetHandler, OLSPostHandler)
var DecTreeHandler = AbstractHandler(DecTreeGetHandler, DecTreePostHandler)
var BaggedHandler = AbstractHandler(BaggedGetHandler, BaggedPostHandler)
var BoostedHandler = AbstractHandler(BoostedGetHandler, BoostedPostHandler)

func StartServer(port string) error {
	// Top level routes
	http.HandleFunc("/models", ModelsHandler)
	http.HandleFunc("/estimators", EstimatorsHandler)
	http.HandleFunc("/ensembles", EnsemblesHandler)

	// Model specific routes
	http.HandleFunc("/models/linreg", LinRegHandler)
	http.HandleFunc("/models/ols", OLSHandler)
	http.HandleFunc("/models/dectree", DecTreeHandler)

	// Ensemble specific routes
	http.HandleFunc("/ensembles/bagged", BaggedHandler)
	http.HandleFunc("/ensembles/boosted", BoostedHandler)

	return http.ListenAndServe(":"+port, nil)
}
