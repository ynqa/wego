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
	"math/rand"

	"github.com/chewxy/lingo/corpus"

	"github.com/ynqa/word-embedding/model"
	"github.com/ynqa/word-embedding/vector"
)

// NegativeSampling is a piece of Word2Vec optimizer.
type NegativeSampling struct {
	negativeTensor     Tensor
	NegativeSampleSize int
}

// NewNegativeSampling creates *NegativeSampling.
// The negative vector is NOT built yet.
func NewNegativeSampling(negativeSampleSize int) *NegativeSampling {
	ns := new(NegativeSampling)
	ns.NegativeSampleSize = negativeSampleSize
	return ns
}

// Init initializes the negative vector.
func (ns *NegativeSampling) Init(c *corpus.Corpus, dimension int) (err error) {
	ns.negativeTensor = NewTensor(c.Size(), dimension)
	return
}

// Update updates the word vector using the negative vector.
func (ns *NegativeSampling) Update(targetID int,
	contextVector, poolVector vector.Vector, learningRate float64) error {

	var label int
	var negativeID int
	var negativeVector vector.Vector

	for n := -1; n < ns.NegativeSampleSize; n++ {
		if n == -1 {
			label = 1
			negativeVector = ns.negativeTensor[targetID]
		} else {
			label = 0
			negativeID := rand.Intn(len(ns.negativeTensor))
			negativeVector = ns.negativeTensor[negativeID]
			if targetID == negativeID {
				continue
			}
		}

		inner := negativeVector.Inner(contextVector)
		f := model.Sigmoid(inner)
		g := (float64(label) - f) * learningRate

		poolVector.UnsafeAdd(negativeVector.Mul(g))
		negativeVector.UnsafeAdd(contextVector.Mul(g))

		if n == -1 {
			ns.negativeTensor[targetID] = negativeVector
		} else {
			ns.negativeTensor[negativeID] = negativeVector
		}
	}
	return nil
}
