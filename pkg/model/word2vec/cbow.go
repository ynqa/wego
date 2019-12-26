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
	"github.com/ynqa/wego/pkg/model"
)

// Cbow behaviors as one of Word2vec solver.
type Cbow struct {
	sums, pools chan []float64

	dimension int
	window    int
}

// NewCbow creates *Cbow
func NewCbow(dimension, window, threadSize int) *Cbow {
	pools := make(chan []float64, threadSize)
	sums := make(chan []float64, threadSize)
	for i := 0; i < threadSize; i++ {
		pools <- make([]float64, dimension)
		sums <- make([]float64, dimension)
	}
	return &Cbow{
		sums:  sums,
		pools: pools,

		dimension: dimension,
		window:    window,
	}
}

func (c *Cbow) trainOne(document []int, wordIndex int, wordVector []float64, lr float64, optimizer Optimizer) {
	sum := <-c.sums
	pool := <-c.pools
	word := document[wordIndex]
	for i := 0; i < c.dimension; i++ {
		sum[i] = 0.0
		pool[i] = 0.0
	}
	c.dowith(document, wordIndex, sum, pool, wordVector, c.initSum)
	optimizer.update(word, lr, sum, pool)
	c.dowith(document, wordIndex, sum, pool, wordVector, c.updateContext)
	c.sums <- sum
	c.pools <- pool
}

func (c *Cbow) dowith(document []int, wordIndex int, sum, pool, wordVector []float64,
	opr func(context int, sum, pool, wordVector []float64)) {

	shrinkage := model.NextRandom(c.window)
	for a := shrinkage; a < c.window*2+1-shrinkage; a++ {
		if a != c.window {
			c := wordIndex - c.window + a
			if c < 0 || c >= len(document) {
				continue
			}
			context := document[c]
			opr(context, sum, pool, wordVector)
		}
	}
}

func (c *Cbow) initSum(context int, sum, pool, wordVector []float64) {
	for i := 0; i < c.dimension; i++ {
		sum[i] += wordVector[context*c.dimension+i]
	}
}

func (c *Cbow) updateContext(context int, sum, pool, wordVector []float64) {
	for i := 0; i < c.dimension; i++ {
		wordVector[context*c.dimension+i] += pool[i]
	}
}
