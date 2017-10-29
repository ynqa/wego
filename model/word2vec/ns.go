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
	"github.com/chewxy/lingo/corpus"
	"math/rand"
)

// NegativeSampling is a piece of Word2Vec optimizer.
type NegativeSampling struct {
	contextVector []float64
	sampleSize    int

	vocabulary int
	dimension  int
}

// NewNegativeSampling creates *NegativeSampling.
// The negative vector is NOT built yet.
func NewNegativeSampling(sampleSize int) *NegativeSampling {
	ns := new(NegativeSampling)
	ns.sampleSize = sampleSize
	return ns
}

// Init initializes the negative vector.
func (ns *NegativeSampling) Init(c *corpus.Corpus, dimension int) (err error) {
	ns.vocabulary = c.Size()
	ns.dimension = dimension

	ns.contextVector = make([]float64, ns.vocabulary*ns.dimension)
	for i := 0; i < ns.vocabulary*ns.dimension; i++ {
		ns.contextVector[i] = (rand.Float64() - 0.5) / float64(ns.dimension)
	}
	return
}

// Update updates the word vector using the negative vector.
func (ns *NegativeSampling) Update(targetID int, contextVector, poolVector []float64, learningRate float64) {

	var label int
	var negativeID int
	var negativeVector []float64

	for n := -1; n < ns.sampleSize; n++ {
		if n == -1 {
			label = 1
			negativeVector = ns.contextVector[targetID*ns.dimension : targetID*ns.dimension+ns.dimension]
		} else {
			label = 0
			negativeID := nextRandom(ns.vocabulary)
			negativeVector = ns.contextVector[negativeID*ns.dimension : negativeID*ns.dimension+ns.dimension]
			if targetID == negativeID {
				continue
			}
		}

		ns.gradUpd(label, learningRate, negativeVector, contextVector, poolVector)

		var index int
		if n == -1 {
			index = targetID
		} else {
			index = negativeID
		}

		for i := 0; i < ns.dimension; i++ {
			ns.contextVector[index*ns.dimension+i] = negativeVector[i]
		}
	}
}

func (ns *NegativeSampling) gradUpd(label int, lr float64, negVec, ctxVec, poolVec []float64) {
	var inner float64
	for i := 0; i < ns.dimension; i++ {
		inner += negVec[i] * ctxVec[i]
	}

	g := (float64(label) - sigmoid(inner)) * lr

	for i := 0; i < ns.dimension; i++ {
		poolVec[i] += g * negVec[i]
		negVec[i] += g * ctxVec[i]
	}
}
