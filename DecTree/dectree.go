package DecTree

import (
	"GoML/Ensemble"
	"GoML/metrics"
	"fmt"
	"math/rand"
	"slices"
	"strings"
	"time"
)

type Node struct {
	featureIndex int
	threshold    float64
	left, right  *Node
	value        float64
	isLeaf       bool
}

type DecTree struct {
	X               [][]float64 `json:"x"`
	Y               []float64   `json:"y"`
	Metrics         metrics.Metrics
	root            *Node
	MaxDepth        int    `json:"max_depth"`
	MinSamplesSplit int    `json:"min_samples_split"`
	MinSamplesLeaf  int    `json:"min_samples_leaf"`
	MaxFeatures     *int   `json:"max_features"`
	RandomSeed      *int64 `json:"random_seed"`
	rng             *rand.Rand
}

func NewDecTree(x [][]float64, y []float64, maxDepth, minSamplesSplit, minSamplesLeaf int, randomSeed *int64, maxFeatures *int) Ensemble.Estimator {
	if randomSeed == nil {
		tNow := time.Now().UnixNano()
		randomSeed = &tNow
	}

	preAllocX := make([][]float64, len(x))
	for i := range x {
		preAllocX[i] = make([]float64, len(x[i]))
		copy(preAllocX[i], x[i])
	}

	preAllocY := make([]float64, len(y))
	copy(preAllocY, y)

	return &DecTree{
		X:               preAllocX,
		Y:               preAllocY,
		MaxDepth:        maxDepth,
		MinSamplesSplit: minSamplesSplit,
		MinSamplesLeaf:  minSamplesLeaf,
		MaxFeatures:     maxFeatures,
		RandomSeed:      randomSeed,
		rng:             rand.New(rand.NewSource(*randomSeed)),
	}
}

func NewDefaultDecTree(x [][]float64, y []float64) Ensemble.Estimator {
	return NewDecTree(x, y, 10, 2, 1, nil, nil)
}

func (dt *DecTree) createLeaf(indices []int) *Node {
	sum := 0.0
	for _, idx := range indices {
		sum += dt.Y[idx]
	}
	return &Node{
		value:  sum / float64(len(indices)),
		isLeaf: true,
	}
}

func (dt *DecTree) splitIndices(indices []int, featureIdx int, threshold float64) (leftIdx, rightIdx []int) {
	for _, idx := range indices {
		if dt.X[idx][featureIdx] <= threshold {
			leftIdx = append(leftIdx, idx)
		} else {
			rightIdx = append(rightIdx, idx)
		}
	}
	return leftIdx, rightIdx
}

func (dt *DecTree) varianceReduction(y []float64, leftIdx, rightIdx []int) float64 {
	totVar := varianceSubset(y, append(leftIdx, rightIdx...))
	leftVar := varianceSubset(y, leftIdx)
	rightVar := varianceSubset(y, rightIdx)

	weightedVar := (float64(len(leftIdx))*leftVar + float64(len(rightIdx))*rightVar) / float64(len(leftIdx)+len(rightIdx))
	return totVar - weightedVar
}

func (dt *DecTree) bestSplit(indices []int) (bestFeature int, bestThreshold float64, bestScore float64) {
	bestFeature = -1
	bestScore = -1.0

	nFeatures := len(dt.X[0])
	m := nFeatures
	if dt.MaxFeatures != nil && *dt.MaxFeatures < nFeatures {
		m = *dt.MaxFeatures
	}

	allFeatures := make([]int, nFeatures)
	for i := 0; i < nFeatures; i++ {
		allFeatures[i] = i
	}
	dt.rng.Shuffle(nFeatures, func(i int, j int) {
		allFeatures[i], allFeatures[j] = allFeatures[j], allFeatures[i]
	})
	features := allFeatures[:m]

	for _, f := range features {
		values := make([]float64, len(indices))
		for i, idx := range indices {
			values[i] = dt.X[idx][f]
		}

		uniques := sortedUnique(values)
		if len(uniques) == 1 {
			continue
		}

		for i := 0; i < len(uniques)-1; i++ {
			threshold := (uniques[i] + uniques[i+1]) / 2
			leftIdx, rightIdx := dt.splitIndices(indices, f, threshold)
			if len(leftIdx) == 0 || len(rightIdx) == 0 {
				continue
			}

			score := dt.varianceReduction(dt.Y, leftIdx, rightIdx)
			if score > bestScore {
				bestScore = score
				bestFeature = f
				bestThreshold = threshold
			}
		}
	}
	return
}

func (dt *DecTree) buildTree(indices []int, depth int) *Node {
	if depth >= dt.MaxDepth || len(indices) < dt.MinSamplesSplit {
		return dt.createLeaf(indices)
	}

	featureIdx, threshold, _ := dt.bestSplit(indices)
	if featureIdx == -1 {
		return dt.createLeaf(indices)
	}

	leftIdx, rightIdx := dt.splitIndices(indices, featureIdx, threshold)
	if len(leftIdx) < dt.MinSamplesLeaf || len(rightIdx) < dt.MinSamplesLeaf {
		return dt.createLeaf(indices)
	}

	node := &Node{
		featureIndex: featureIdx,
		threshold:    threshold,
	}
	node.left = dt.buildTree(leftIdx, depth+1)
	node.right = dt.buildTree(rightIdx, depth+1)

	return node
}

func sortedUnique(input []float64) []float64 {
	isUnique := make(map[float64]bool)
	for _, v := range input {
		isUnique[v] = true
	}
	unique := make([]float64, 0, len(isUnique))
	for k := range isUnique {
		unique = append(unique, k)
	}
	slices.Sort(unique)
	return unique
}

func varianceSubset(y []float64, indices []int) float64 {
	if len(indices) == 0 {
		return 0.0
	}
	mean := 0.0
	for _, idx := range indices {
		mean += y[idx]
	}
	mean /= float64(len(indices))

	variance := 0.0
	for _, idx := range indices {
		variance += (y[idx] - mean) * (y[idx] - mean)
	}
	return variance / float64(len(indices))
}

func (dt *DecTree) Fit() {
	indices := make([]int, len(dt.Y))
	for i := range dt.Y {
		indices[i] = i
	}
	dt.root = dt.buildTree(indices, 0)

	preds := make([]float64, len(dt.Y))
	for i, row := range dt.X {
		preds[i] = dt.Predict(row)
	}
	dt.Metrics = metrics.Evaluate(dt.Y, preds)
}

func (dt *DecTree) Predict(x []float64) float64 {
	node := dt.root
	for !node.isLeaf {
		if x[node.featureIndex] <= node.threshold {
			node = node.left
		} else {
			node = node.right
		}
	}
	return node.value
}

func (dt *DecTree) GetMetrics() metrics.Metrics {
	return dt.Metrics
}

func (dt *DecTree) GetTreeString() string {
	var buildString func(node *Node, depth int) string
	buildString = func(node *Node, depth int) string {
		if node.isLeaf {
			return fmt.Sprintf("%sLeaf: %.4f\n", strings.Repeat("  ", depth), node.value)
		}
		leftStr := buildString(node.left, depth+1)
		rightStr := buildString(node.right, depth+1)
		return fmt.Sprintf("%s[Feature %d <= %.4f]\n%s%s", strings.Repeat("  ", depth), node.featureIndex, node.threshold, leftStr, rightStr)
	}
	return buildString(dt.root, 0)
}

func (dt *DecTree) GetFeatureImportance() map[int]float64 {
	importance := make(map[int]float64)
	var traverse func(node *Node)
	traverse = func(node *Node) {
		if node == nil || node.isLeaf {
			return
		}
		importance[node.featureIndex] += 1.0
		traverse(node.left)
		traverse(node.right)
	}
	traverse(dt.root)

	// Normalize importance
	total := 0.0
	for _, v := range importance {
		total += v
	}
	for k := range importance {
		importance[k] /= total
	}
	return importance
}
