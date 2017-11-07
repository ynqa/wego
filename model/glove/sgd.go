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
	"github.com/ynqa/word-embedding/model"
)

// SGD behaviors as one of GloVe solver.
type SGD struct {
	dimension int

	currentLearningRate float64
	shrinkage           float64
}

// NewSGD creates *SGD.
func NewSGD(c *model.Config) *SGD {
	return &SGD{
		dimension: c.Dimension,

		currentLearningRate: c.InitLearningRate,
		shrinkage:           0.9,
	}
}

func (s *SGD) init(weightSize int) {}

func (s *SGD) trainOne(l1, l2 int, f, coefficient float64, weight []float64) float64 {
	var diff, cost float64
	for i := 0; i < s.dimension; i++ {
		diff += weight[l1+i] * weight[l2+i]
	}
	diff += weight[l1+s.dimension] + weight[l2+s.dimension] - f
	fdiff := diff * coefficient
	cost = 0.5 * fdiff * diff
	fdiff *= s.currentLearningRate

	for i := 0; i < s.dimension; i++ {
		temp1 := fdiff * weight[l2+i]
		temp2 := fdiff * weight[l1+i]
		weight[l1+i] -= temp1
		weight[l2+i] -= temp2
	}

	weight[l1+s.dimension] -= fdiff
	weight[l2+s.dimension] -= fdiff

	return cost
}

func (s *SGD) callback() {
	s.currentLearningRate *= s.shrinkage
}
