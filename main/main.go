package main

import (
	"GoML/demo"
	"GoML/httpServer"
	"GoML/parser"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func flowUsage() {
	fmt.Println("Usage:")
	fmt.Println("model names: linreg, ols, dectree")
	fmt.Println("ensemble methods: bagged, boosted")
	fmt.Println("nEstimators must be an integer (>0)")
	fmt.Println("Accepted y/n inputs (case insensitive): y, yes, n, no")

	os.Exit(-1)
}

func panicUsage(usage func()) {
	usage()
	os.Exit(-1)
}

var models = map[string]struct{}{
	"linreg":  {},
	"ols":     {},
	"dectree": {},
}

var ensembles = map[string]struct{}{
	"bagged":  {},
	"boosted": {},
}

func mainLoop(filePath string, hasHeaders bool, targetIndex int) {
	var err error
	data := parser.LoadData(filePath, ",", hasHeaders, targetIndex)
	dummyX, dummyY := data.X, data.Y
	fmt.Printf("Data Loaded: %d samples, %d features\n", len(dummyX), len(dummyX[0]))
	fmt.Printf("Feature Names: %v\n", data.FeatureNames)
	fmt.Printf("Target Name: %s\n", data.TargetName)

	var runDemo string
	fmt.Println("Run model demo? (y/n): ")
	_, err = fmt.Scanln(&runDemo)
	if err != nil {
		panicUsage(flag.Usage)
	}
	if strings.ToLower(runDemo) == "y" || strings.ToLower(runDemo) == "yes" {
		var modelName string
		var ensembleYN string
		var isEnsemble bool
		var ensembleMethod string
		var nEstimators int
		var reRun string

		fmt.Println("Enter Model Name (linreg, ols, dectree): ")
		_, err = fmt.Scanln(&modelName)
		if err != nil || strings.TrimSpace(modelName) == "" {
			panicUsage(flowUsage)
		}
		if _, ok := models[strings.ToLower(modelName)]; !ok {
			panicUsage(flowUsage)
		}

		fmt.Println("Use Ensemble? (y/n): ")
		_, err := fmt.Scanln(&ensembleYN)
		if err != nil || strings.TrimSpace(ensembleYN) == "" {
			panicUsage(flowUsage)
		}
		if strings.ToLower(ensembleYN) == "y" || strings.ToLower(ensembleYN) == "yes" {
			isEnsemble = true
			fmt.Println("Enter Ensemble Method (bagged, boosted): ")
			_, err = fmt.Scanln(&ensembleMethod)
			if err != nil || strings.TrimSpace(ensembleMethod) == "" {
				panicUsage(flowUsage)
			}
			if _, ok := ensembles[strings.ToLower(ensembleMethod)]; !ok {
				panicUsage(flowUsage)
			}

			fmt.Println("Enter Number of Estimators (>0): ")
			_, err = fmt.Scanln(&nEstimators)
			if err != nil {
				panicUsage(flowUsage)
			}
			if nEstimators <= 0 {
				panicUsage(flowUsage)
			}
		} else {
			isEnsemble = false
			ensembleMethod = ""
			nEstimators = 0
		}
		demo.Run(dummyX, dummyY, modelName, isEnsemble, ensembleMethod, nEstimators)

		fmt.Println("Run another model demo? (y/n): ")
		_, err = fmt.Scanln(&reRun)
		if err != nil {
			panicUsage(flowUsage)
		}
		if strings.ToLower(reRun) == "y" || strings.ToLower(reRun) == "yes" {
			mainLoop(filePath, hasHeaders, targetIndex)
		} else {
			fmt.Println("Exiting...")
			os.Exit(0)
		}
	}

}

func main() {
	flag.Usage = func() {
		fmt.Println("Usage with Flags:")
		flag.PrintDefaults()
		fmt.Println("\nUsage with Args:")
		fmt.Println("go run main.go <string file_path> [<bool hasHeaders>] <int target_index>")
	}
	var demoFlag = flag.Bool("serve", false, "<bool> Serve HTTP demo over localhost. All other flags will be ignored if true. (default false)")
	var filePathFlag = flag.String("data-csv", "", "<string> Path to CSV data file")
	var hasHeadersFlag = flag.Bool("h", false, "<bool> Whether the CSV file has headers")
	var targetIndexFlag = flag.Int("target-index", -1, "<int> Index of the target column (0-based)")

	var filePath string
	var hasHeaders bool
	var targetIndex int
	var err error
	flag.Parse()

	if *demoFlag {
		fmt.Println("Starting HTTP server on http://localhost:8080 ...")
		err := httpServer.StartServer("8080")
		if err != nil {
			panic(err)
		}
	}

	args := flag.Args()
	switch len(args) {
	case 2:
		filePath = args[0]
		if strings.TrimSpace(filePath) == "" {
			panicUsage(flag.Usage)
		}

		targetIndex, err = strconv.Atoi(args[1])
		if err != nil {
			panicUsage(flag.Usage)
		}

		hasHeaders = false
	case 3:
		filePath = args[0]
		if strings.TrimSpace(filePath) == "" {
			panicUsage(flag.Usage)
		}

		hasHeaders, err = strconv.ParseBool(strings.ToLower(args[1]))
		if err != nil {
			panicUsage(flag.Usage)
		}

		targetIndex, err = strconv.Atoi(args[2])
		if err != nil {
			panicUsage(flag.Usage)
		}

	default:
		filePath = *filePathFlag
		if strings.TrimSpace(filePath) == "" {
			panicUsage(flag.Usage)
		}

		hasHeaders = *hasHeadersFlag
		if hasHeaders != true && hasHeaders != false {
			panicUsage(flag.Usage)
		}

		targetIndex = *targetIndexFlag
		if targetIndex < 0 {
			panicUsage(flag.Usage)
		}
	}

	mainLoop(filePath, hasHeaders, targetIndex)
}
