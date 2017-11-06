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
	"github.com/ynqa/word-embedding/model"
)

// CBOW is a piece of Word2Vec model.
type CBOW struct {
	sums, pools chan []float64

	window    int
	dimension int
}

// NewCBOW creates *CBOW
func NewCBOW(c *model.Config) *CBOW {
	pools := make(chan []float64, c.Thread)
	sums := make(chan []float64, c.Thread)

	for i := 0; i < c.Thread; i++ {
		pools <- make([]float64, c.Dimension)
		sums <- make([]float64, c.Dimension)
	}

	return &CBOW{
		sums:  sums,
		pools: pools,

		window:    c.Window,
		dimension: c.Dimension,
	}
}

func (c *CBOW) trainOne(wordIDs []int, wordIndex int, wordVector []float64, lr float64, optimizer Optimizer) {
	sum := <-c.sums
	pool := <-c.pools

	targetID := wordIDs[wordIndex]

	for i := 0; i < c.dimension; i++ {
		sum[i] = 0.0
		pool[i] = 0.0
	}
	c.dowith(wordIDs, wordIndex, sum, pool, wordVector, c.initSum)

	optimizer.update(targetID, sum, pool, lr)

	c.dowith(wordIDs, wordIndex, sum, pool, wordVector, c.updateCtx)

	c.sums <- sum
	c.pools <- pool
}

func (c *CBOW) dowith(wordIDs []int, wordIndex int, sum, pool, wordVector []float64,
	g func(contextID int, sum, pool, wordVector []float64)) {

	shr := nextRandom(c.window)
	for a := shr; a < c.window*2+1-shr; a++ {
		if a != c.window {
			c := wordIndex - c.window + a
			if c < 0 || c >= len(wordIDs) {
				continue
			}
			contextID := wordIDs[c]
			g(contextID, sum, pool, wordVector)
		}
	}
}

func (c *CBOW) initSum(contextID int, sum, pool, wordVector []float64) {
	for i := 0; i < c.dimension; i++ {
		sum[i] += wordVector[contextID*c.dimension+i]
	}
}

func (c *CBOW) updateCtx(contextID int, sum, pool, wordVector []float64) {
	for i := 0; i < c.dimension; i++ {
		wordVector[contextID*c.dimension+i] += pool[i]
	}
}
