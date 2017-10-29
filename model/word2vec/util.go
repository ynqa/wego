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

package word2vec

import (
	"math"
	"strings"
)

var next uint64 = 1

// Linear congruential generator like rand.Intn(window)
func nextRandom(value int) int {
	next = next*uint64(25214903917) + 11
	return int(next % uint64(value))
}

func lower(a []string) {
	for i := range a {
		a[i] = strings.ToLower(a[i])
	}
}

// sigmoid returns f(x) = \frac{1}{1 + e^{-x}}.
// See: http://en.wikipedia.org/wiki/Sigmoid_function.
func sigmoid(f float64) float64 {
	exp := math.Exp(f)
	if math.IsInf(exp, 1) {
		return 1.0
	}
	return exp / (1.0 + exp)
}
