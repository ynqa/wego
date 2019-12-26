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

package glove

import (
	"math"
)

// AdaGrad behaviors as one of Glove solver.
type AdaGrad struct {
	dimension int
	initlr    float64
	gradsq    []float64
}

// NewAdaGrad creates *AdaGrad.
func NewAdaGrad(dimension int, initlr float64) *AdaGrad {
	return &AdaGrad{
		dimension: dimension,
		initlr:    initlr,
	}
}

func (a *AdaGrad) initialize(vectorSize int) {
	a.gradsq = make([]float64, vectorSize)
	for i := 0; i < vectorSize; i++ {
		a.gradsq[i] = 1.
	}
}

func (a *AdaGrad) trainOne(l1, l2 int, f, coefficient float64, vector []float64) float64 {
	var diff, cost float64
	for i := 0; i < a.dimension; i++ {
		diff += vector[l1+i] * vector[l2+i]
	}
	diff += vector[l1+a.dimension] + vector[l2+a.dimension] - f
	fdiff := diff * coefficient
	cost = 0.5 * fdiff * diff
	fdiff *= a.initlr
	for i := 0; i < a.dimension; i++ {
		temp1 := fdiff * vector[l2+i]
		temp2 := fdiff * vector[l1+i]
		a.gradsq[l1+i] += temp1 * temp1
		a.gradsq[l2+i] += temp2 * temp2

		temp1 /= math.Sqrt(a.gradsq[l1+i])
		temp2 /= math.Sqrt(a.gradsq[l2+i])
		vector[l1+i] -= temp1
		vector[l2+i] -= temp2
	}
	vector[l1+a.dimension] -= fdiff / math.Sqrt(a.gradsq[l1+a.dimension])
	vector[l2+a.dimension] -= fdiff / math.Sqrt(a.gradsq[l2+a.dimension])
	fdiff *= fdiff
	a.gradsq[l1+a.dimension] += fdiff
	a.gradsq[l2+a.dimension] += fdiff
	return cost
}
