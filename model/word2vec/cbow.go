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
	"io"
)

// CBOW is a piece of Word2Vec model.
type CBOW struct {
	*State

	sums, pools chan []float64
}

// NewCBOW creates *CBOW
func NewCBOW(s *State) *CBOW {
	pools := make(chan []float64, s.Thread)
	sums := make(chan []float64, s.Thread)

	for i := 0; i < s.Thread; i++ {
		pools <- make([]float64, s.Dimension)
		sums <- make([]float64, s.Dimension)
	}

	return &CBOW{
		State: s,
		sums:  sums,
		pools: pools,
	}
}

// Train call Trainer with CBOW trainOne.
func (c *CBOW) Train(f io.ReadCloser) error {
	return c.Trainer(f, c.trainOne)
}

func (c *CBOW) trainOne(wordIDs []int, wordIndex int, lr float64) {
	sum := <-c.sums
	pool := <-c.pools

	targetID := wordIDs[wordIndex]

	for i := 0; i < c.Dimension; i++ {
		sum[i] = 0.0
		pool[i] = 0.0
	}
	c.dowith(wordIDs, wordIndex, sum, pool, c.initSum)

	c.opt.Update(targetID, sum, pool, lr)

	c.dowith(wordIDs, wordIndex, sum, pool, c.updateCtx)

	c.sums <- sum
	c.pools <- pool
}

func (c *CBOW) dowith(wordIDs []int, wordIndex int, sum, pool []float64, g func(contextID int, sum, pool []float64)) {
	shr := nextRandom(c.Window)
	for a := shr; a < c.Window*2+1-shr; a++ {
		if a != c.Window {
			c := wordIndex - c.Window + a
			if c < 0 || c >= len(wordIDs) {
				continue
			}
			contextID := wordIDs[c]
			g(contextID, sum, pool)
		}
	}
}

func (c *CBOW) initSum(contextID int, sum, pool []float64) {
	for i := 0; i < c.Dimension; i++ {
		sum[i] += c.vector[contextID*c.Dimension+i]
	}
}

func (c *CBOW) updateCtx(contextID int, sum, pool []float64) {
	for i := 0; i < c.Dimension; i++ {
		c.vector[contextID*c.Dimension+i] += pool[i]
	}
}
