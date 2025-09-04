package parser

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

type CSVData struct {
	Header []string
	Rows   [][]float64
}

type DataSet struct {
	X            [][]float64
	Y            []float64
	FeatureNames []string
	TargetName   string
}

func isASCII(buff []byte) bool {
	for _, b := range buff {
		if b > unicode.MaxASCII && b != byte('"') {
			return false
		}
	}
	return true
}

func GetNumericRow(s string, sep string) (row []float64) {
	raw := strings.Split(s, sep)
	row = make([]float64, len(raw))
	for i, r := range raw {
		r = strings.TrimSpace(r)
		r = strings.Trim(r, `"`)
		val, err := strconv.ParseFloat(r, 64)
		if err != nil {
			panic(err)
		}
		row[i] = val
	}
	return
}

func splitRows(s string) []string {
	return strings.Split(strings.TrimSpace(s), "\n")
}

func ParseCSV(filePath string, sep string, hasHeader bool) CSVData {
	data, err := os.ReadFile(filePath)
	out := CSVData{}

	if err != nil {
		panic(err)
	}
	if !isASCII(data) {
		panic("File is not ASCII encoded")
	}

	split := splitRows(string(data))
	if hasHeader {
		out.Header = strings.Split(split[0], sep)
		split = split[1:]
	} else {
		nCols := len(strings.Split(split[0], sep))
		header := make([]string, nCols)
		for i := 0; i < nCols; i++ {
			header[i] = fmt.Sprintf("col_%d", i)
		}
		out.Header = header
	}
	out.Rows = make([][]float64, 0, len(split))
	for i, row := range split {
		if strings.TrimSpace(row) == "" {
			continue
		}
		numericRow := GetNumericRow(row, sep)
		if len(numericRow) != len(out.Header) {
			panic(fmt.Sprintf("Row %d has %d columns, expected %d", i, len(numericRow), len(out.Header)))
		}
		out.Rows = append(out.Rows, numericRow)
	}

	return out
}

func LoadData(filePath string, sep string, hasHeader bool, targetCol int) DataSet {
	csvData := ParseCSV(filePath, sep, hasHeader)
	out := DataSet{}
	if targetCol < 0 || targetCol >= len(csvData.Header) {
		panic("Invalid target column index")
	}

	nRows := len(csvData.Rows)
	nCols := len(csvData.Header)
	X := make([][]float64, nRows)
	Y := make([]float64, nRows)
	featureNames := make([]string, 0, nCols-1)
	targetName := csvData.Header[targetCol]

	for i, name := range csvData.Header {
		if i != targetCol {
			featureNames = append(featureNames, name)
		}
	}

	for i, row := range csvData.Rows {
		target := row[targetCol]
		Y[i] = target
		features := make([]float64, 0, nCols-1)
		features = slices.Delete(row, targetCol, targetCol+1)
		X[i] = features
	}
	out.X = X
	out.Y = Y
	out.TargetName = targetName
	out.FeatureNames = featureNames

	return out

}
