// Copyright © 2017 Makoto Ito
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"math"
)

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

var next uint64 = 1

// NextRandom is linear congruential generator like rand.Intn(window)
func NextRandom(value int) int {
	next = next*uint64(25214903917) + 11
	return int(next % uint64(value))
}
