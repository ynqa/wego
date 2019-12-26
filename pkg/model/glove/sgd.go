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

// Sgd is stochastic gradient descent that behaviors as one of GloVe solver.
type Sgd struct {
	dimension int
	currentlr float64
	shrinkage float64
}

// NewSgd creates *Sgd.
func NewSgd(dimension int, initlr float64) *Sgd {
	return &Sgd{
		dimension: dimension,
		currentlr: initlr,
		shrinkage: 0.9,
	}
}

func (s *Sgd) trainOne(l1, l2 int, f, coefficient float64, vector []float64) float64 {
	var diff, cost float64
	for i := 0; i < s.dimension; i++ {
		diff += vector[l1+i] * vector[l2+i]
	}
	diff += vector[l1+s.dimension] + vector[l2+s.dimension] - f
	fdiff := diff * coefficient
	cost = 0.5 * fdiff * diff
	fdiff *= s.currentlr
	for i := 0; i < s.dimension; i++ {
		temp1 := fdiff * vector[l2+i]
		temp2 := fdiff * vector[l1+i]
		vector[l1+i] -= temp1
		vector[l2+i] -= temp2
	}
	vector[l1+s.dimension] -= fdiff
	vector[l2+s.dimension] -= fdiff
	return cost
}

func (s *Sgd) postOneIter() {
	s.currentlr *= s.shrinkage
}
