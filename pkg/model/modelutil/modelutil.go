package modelutil

import (
	"math"
)

var (
	next uint64 = 1
)

// NextRandom is linear congruential generator (rand.Intn).
func NextRandom(value int) int {
	next = next*uint64(25214903917) + 11
	return int(next % uint64(value))
}

// IndexPerThread creates interval of indices per thread.
func IndexPerThread(threadSize, dataSize int) []int {
	indexPerThread := make([]int, threadSize+1)
	indexPerThread[0] = 0
	indexPerThread[threadSize] = dataSize
	for i := 1; i < threadSize; i++ {
		indexPerThread[i] = indexPerThread[i-1] +
			int(math.Trunc(float64((dataSize+i)/threadSize)))
	}
	return indexPerThread
}
