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

package co

import (
	"math"
)

// The data structure for co-occurrence is referred from:
//   https://blog.chewxy.com/2017/07/12/21-bits-english/

// EncodeBigram creates id between two words.
func EncodeBigram(l1, l2 uint64) uint64 {
	return l1 | (l2 << 32)
}

// DecodeBigram reverts pair id to two word ids.
func DecodeBigram(pid uint64) (uint64, uint64) {
	f := pid >> 32
	return pid - (f << 32), f
}

// Inc is the function for increment co-occurrence.
func Inc(l1, l2 int) float64 {
	return 1.
}

// IncDist is the function for increment co-occurrence with scaled by distance.
func IncDist(l1, l2 int) float64 {
	return 1. / math.Abs(float64(l1-l2))
}
