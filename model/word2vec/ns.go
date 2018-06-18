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
	"github.com/ynqa/wego/corpus"
	"github.com/ynqa/wego/model"
)

// NegativeSampling is a piece of Word2Vec optimizer.
type NegativeSampling struct {
	*SigmoidTable
	contextVector []float64
	sampleSize    int

	dimension  int
	vocabulary int
}

// NewNegativeSampling creates *NegativeSampling.
func NewNegativeSampling(sampleSize int) *NegativeSampling {
	ns := new(NegativeSampling)
	ns.SigmoidTable = newSigmoidTable()
	ns.sampleSize = sampleSize
	return ns
}

func (ns *NegativeSampling) initialize(cps *corpus.Word2vecCorpus, dimension int) error {
	ns.vocabulary = cps.Size()
	ns.dimension = dimension
	ns.contextVector = make([]float64, ns.vocabulary*ns.dimension)
	return nil
}

func (ns *NegativeSampling) update(word int, lr float64, vector, poolVector []float64) {
	var label int
	var sample int
	var sampleVector []float64
	for n := -1; n < ns.sampleSize; n++ {
		if n == -1 {
			label = 1
			sampleVector = ns.contextVector[word*ns.dimension : word*ns.dimension+ns.dimension]
		} else {
			label = 0
			sample = model.NextRandom(ns.vocabulary)
			sampleVector = ns.contextVector[sample*ns.dimension : sample*ns.dimension+ns.dimension]
			if word == sample {
				continue
			}
		}
		ns.gradUpd(label, lr, sampleVector, vector, poolVector)
		var index int
		if n == -1 {
			index = word
		} else {
			index = sample
		}
		for i := 0; i < ns.dimension; i++ {
			ns.contextVector[index*ns.dimension+i] = sampleVector[i]
		}
	}
}

func (ns *NegativeSampling) gradUpd(label int, lr float64, sampledVector, vector, poolVector []float64) {
	var inner float64
	for i := 0; i < ns.dimension; i++ {
		inner += sampledVector[i] * vector[i]
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
		poolVector[i] += g * sampledVector[i]
		sampledVector[i] += g * vector[i]
	}
}
