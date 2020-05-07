package searchutil

import (
	"math"
)

func Norm(vec []float64) float64 {
	var n float64
	for _, v := range vec {
		n += math.Pow(v, 2)
	}
	return math.Sqrt(n)
}

func Cosine(v1, v2 []float64, n1, n2 float64) float64 {
	if n1 == 0 || n2 == 0 {
		return 0
	}
	var dot float64
	for i := range v1 {
		dot += v1[i] * v2[i]
	}
	return dot / n1 / n2
}
