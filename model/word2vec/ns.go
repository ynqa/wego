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
	"github.com/ynqa/word-embedding/corpus"
)

// NegativeSampling is a piece of Word2Vec optimizer.
type NegativeSampling struct {
	*SigmoidTable
	contextVector []float64
	sampleSize    int

	vocabulary int
	dimension  int
}

// NewNegativeSampling creates *NegativeSampling.
// The negative vector is NOT built yet.
func NewNegativeSampling(sampleSize int) *NegativeSampling {
	ns := new(NegativeSampling)
	ns.SigmoidTable = newSigmoidTable()
	ns.sampleSize = sampleSize
	return ns
}

func (ns *NegativeSampling) initialize(c *corpus.PredictModelCorpus, dimension int) (err error) {
	ns.vocabulary = c.Size()
	ns.dimension = dimension
	ns.contextVector = make([]float64, ns.vocabulary*ns.dimension)
	return
}

func (ns *NegativeSampling) update(targetID int, contextVector, poolVector []float64, learningRate float64) {

	var label int
	var ctxID int
	var ctxVector []float64

	for n := -1; n < ns.sampleSize; n++ {
		if n == -1 {
			label = 1
			ctxVector = ns.contextVector[targetID*ns.dimension : targetID*ns.dimension+ns.dimension]
		} else {
			label = 0
			ctxID = nextRandom(ns.vocabulary)
			ctxVector = ns.contextVector[ctxID*ns.dimension : ctxID*ns.dimension+ns.dimension]
			if targetID == ctxID {
				continue
			}
		}

		ns.gradUpd(label, learningRate, ctxVector, contextVector, poolVector)

		var index int
		if n == -1 {
			index = targetID
		} else {
			index = ctxID
		}

		for i := 0; i < ns.dimension; i++ {
			ns.contextVector[index*ns.dimension+i] = ctxVector[i]
		}
	}
}

func (ns *NegativeSampling) gradUpd(label int, lr float64, ctxVector, contextVector, poolVector []float64) {
	var inner float64
	for i := 0; i < ns.dimension; i++ {
		inner += ctxVector[i] * contextVector[i]
	}

	var g float64
	if inner <= -ns.maxExp {
		g = (float64(label - 0)) * lr
	} else if inner >= ns.maxExp {
		g = (float64(label - 1)) * lr
	} else {
		g = (float64(label) - ns.sigmoid(inner)) * lr
	}

	for i := 0; i < ns.dimension; i++ {
		poolVector[i] += g * ctxVector[i]
		ctxVector[i] += g * contextVector[i]
	}
}
