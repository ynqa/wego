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

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/lingo/corpus"
	"github.com/chewxy/word-embedding/model"
	"github.com/pkg/errors"
)

// NegativeSampling is a piece of Word2Vec optimizer.
type NegativeSampling struct {
	negativeTensor     *Embedding
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
	ns.negativeTensor = newEmbedding(c.Size(), dimension)
	return
}

// Update updates the word vector using the negative vector.
func (ns *NegativeSampling) Update(targetID int, contextVector, poolVector tensor.Tensor, learningRate float64) error {

	var label float64
	var negativeID int
	var negativeVector *model.SyncTensor

	for n := -1; n < ns.NegativeSampleSize; n++ {
		if n == -1 {
			label = 1
			negativeVector = ns.negativeTensor.m[targetID]
		} else {
			label = 0
			negativeID := rand.Intn(len(ns.negativeTensor.m))
			negativeVector = ns.negativeTensor.m[negativeID]
			if targetID == negativeID {
				continue
			}
		}

		if err := ns.gradUpd(label, learningRate, negativeVector, contextVector, poolVector); err != nil {
			return errors.Wrap(err, "gradUpdate failed for NS")
		}
		// inner, _ := tensor.Inner(negativeVector, contextVector)
		// g := ns.gradUpd(label, learningRate, inner)
		// f := model.Sigmoid(inner)
		// g := (float64(label) - f) * learningRate

		// tensor.FMA(negativeVector, g, poolVector)
		// tensor.FMA(contextVector, g, negativeVector)
		// poolVector.UnsafeAdd(negativeVector.Mul(g))
		// negativeVector.UnsafeAdd(contextVector.Mul(g))

		if n == -1 {
			ns.negativeTensor.m[targetID] = negativeVector
		} else {
			ns.negativeTensor.m[negativeID] = negativeVector
		}
	}
	return nil
}

func (ns *NegativeSampling) gradUpd(label, lr float64, negVec *model.SyncTensor, ctxVec, poolVec tensor.Tensor) (err error) {
	negVec.Lock()
	defer negVec.Unlock()

	switch negVec.Dtype() {
	case tensor.Float64:
		var inner float64
		if ip, ok := eng.(tensor.InnerProderF64); ok {
			if inner, err = ip.Inner(negVec.Tensor, ctxVec); err != nil {
				return errors.Wrap(err, "Inner failed")
			}
		} else {
			return errors.New("Engine does not perform Inner for Float64")
		}
		sig := model.SigmoidF64(inner)
		g := (label - sig) * lr
		tensor.FMA(negVec, &g, poolVec)
		tensor.FMA(ctxVec, &g, negVec)
	case tensor.Float32:
		var inner float32
		if ip, ok := eng.(tensor.InnerProderF32); ok {
			if inner, err = ip.Inner(negVec.Tensor, ctxVec); err != nil {
				return errors.Wrap(err, "Inner failed")
			}
		} else {
			return errors.New("Engine does not perform Inner for Float64")
		}
		sig := model.SigmoidF32(inner)
		g := (float32(label) - sig) * float32(lr)
		tensor.FMA(negVec, &g, poolVec)
		tensor.FMA(ctxVec, &g, negVec)
	}

	return nil
}
