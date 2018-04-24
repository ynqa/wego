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

	"github.com/ynqa/word-embedding/model"
)

// AdaGrad behaviors as one of GloVe solver.
type AdaGrad struct {
	dimension int

	initLearningRate float64
	gradsq           []float64
}

// NewAdaGrad creates *AdaGrad.
func NewAdaGrad(c *model.Config) *AdaGrad {
	return &AdaGrad{
		dimension: c.Dimension,

		initLearningRate: c.InitLearningRate,
	}
}

func (a *AdaGrad) init(weightSize int) {
	a.gradsq = make([]float64, weightSize)

	for i := 0; i < weightSize; i++ {
		a.gradsq[i] = 1.
	}
}

func (a *AdaGrad) trainOne(l1, l2 int, f, coefficient float64, weight []float64) float64 {
	var diff, cost float64
	for i := 0; i < a.dimension; i++ {
		diff += weight[l1+i] * weight[l2+i]
	}
	diff += weight[l1+a.dimension] + weight[l2+a.dimension] - f
	fdiff := diff * coefficient
	cost = 0.5 * fdiff * diff
	fdiff *= a.initLearningRate

	for i := 0; i < a.dimension; i++ {
		temp1 := fdiff * weight[l2+i]
		temp2 := fdiff * weight[l1+i]
		a.gradsq[l1+i] += temp1 * temp1
		a.gradsq[l2+i] += temp2 * temp2

		temp1 /= math.Sqrt(a.gradsq[l1+i])
		temp2 /= math.Sqrt(a.gradsq[l2+i])
		weight[l1+i] -= temp1
		weight[l2+i] -= temp2
	}

	weight[l1+a.dimension] -= fdiff / math.Sqrt(a.gradsq[l1+a.dimension])
	weight[l2+a.dimension] -= fdiff / math.Sqrt(a.gradsq[l2+a.dimension])
	fdiff *= fdiff
	a.gradsq[l1+a.dimension] += fdiff
	a.gradsq[l2+a.dimension] += fdiff
	return cost
}

func (a *AdaGrad) postOneIter() {}
