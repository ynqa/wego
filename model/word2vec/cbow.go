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
	"runtime"

	"gorgonia.org/tensor"
)

// CBOW is a piece of Word2Vec model.
type CBOW struct {
	*State

	sums, pools chan tensor.Tensor
}

// NewCBOW creates *CBOW
func NewCBOW(s *State) *CBOW {
	maxprocs := runtime.NumCPU()
	pools := make(chan tensor.Tensor, maxprocs)
	sums := make(chan tensor.Tensor, maxprocs)

	for i := 0; i < maxprocs; i++ {
		pools <- tensor.New(tensor.WithShape(s.Dimension), tensor.Of(s.Type.D), tensor.WithEngine(s.Type.E))
		sums <- tensor.New(tensor.WithShape(s.Dimension), tensor.Of(s.Type.D), tensor.WithEngine(s.Type.E))
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

func (c *CBOW) trainOne(wordIDs []int, wordIndex int, lr float64) error {
	sum := <-c.sums
	pool := <-c.pools

	targetID := wordIDs[wordIndex]
	sum.Zero()
	pool.Zero()
	c.dowith(wordIDs, wordIndex, sum, pool, c.initSum)

	if err := c.opt.Update(c.Type, targetID, sum, pool, lr); err != nil {
		c.sums <- sum
		c.pools <- pool
		return err
	}
	c.dowith(wordIDs, wordIndex, sum, pool, c.updateCtx)

	c.sums <- sum
	c.pools <- pool
	return nil
}

func (c *CBOW) dowith(wordIDs []int, wordIndex int, sum, pool tensor.Tensor, g func(contextID int, sum, pool tensor.Tensor)) {
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

func (c *CBOW) initSum(contextID int, sum, pool tensor.Tensor) {
	tensor.Add(sum, c.emb.m[contextID], tensor.UseUnsafe())
}

func (c *CBOW) updateCtx(contextID int, sum, pool tensor.Tensor) {
	tensor.Add(c.emb.m[contextID], pool, tensor.UseUnsafe())
}
