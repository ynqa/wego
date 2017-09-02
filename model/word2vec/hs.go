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
	"errors"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/lingo/corpus"

	"github.com/chewxy/word-embedding/model"
	"github.com/chewxy/word-embedding/model/word2vec/huffman"
)

// HierarchicalSoftmax is a piece of Word2Vec optimizer.
type HierarchicalSoftmax struct {
	nodeMap  map[int]*huffman.Node
	MaxDepth int
}

// NewHierarchicalSoftmax creates *HierarchicalSoftmax.
// The huffman tree is NOT built yet.
func NewHierarchicalSoftmax(maxDepth int) *HierarchicalSoftmax {
	hs := new(HierarchicalSoftmax)
	hs.MaxDepth = maxDepth
	return hs
}

// Init initializes the huffman tree.
func (hs *HierarchicalSoftmax) Init(c *corpus.Corpus, dimension int) (err error) {
	hs.nodeMap, err = huffman.NewHuffmanTree(c, dimension, dtype, eng)
	return
}

// Update updates the word vector using the huffman tree.
func (hs *HierarchicalSoftmax) Update(targetID int, contextVector, poolVector tensor.Tensor, learningRate float64) error {

	path := hs.nodeMap[targetID].GetPath()
	for p := 0; p < len(path)-1; p++ {
		relayPoint := path[p]

		childCode := path[p+1].Code

		if err := hs.gradUpd(float64(childCode), learningRate, relayPoint.Vector, poolVector, contextVector); err != nil {
			return err
		}
		// inner, _ := tensor.Inner(contextVector, relayPoint.Vector)
		// inner := contextVector.Inner(relayPoint.Vector)
		// g := hs.gradUpd(float64(childCode), learningRate, inner)
		// f := model.Sigmoid(inner)
		// g := (1.0 - float64(childCode) - f) * learningRate

		// tensor.FMA(relayPoint.Vector, g, poolVector)
		// tensor.FMA(contextVector, g, relayPoint.Vector)
		// poolVector.UnsafeAdd(relayPoint.Vector.Mul(g))
		// relayPoint.Vector.UnsafeAdd(contextVector.Mul(g))

		if hs.MaxDepth > 0 && p >= hs.MaxDepth {
			break
		}
	}
	return nil
}

func (hs *HierarchicalSoftmax) gradUpd(childCode, lr float64, relayPointVec, poolVec, ctxVec tensor.Tensor) (err error) {
	switch relayPointVec.Dtype() {
	case tensor.Float64:
		var inner float64
		if ip, ok := eng.(tensor.InnerProderF64); ok {
			if inner, err = ip.Inner(ctxVec, relayPointVec); err != nil {
				return
			}
		} else {
			return errors.New("Engine does not perform Inner for Float64")
		}

		sig := model.SigmoidF64(inner)
		g := 1.0 - childCode - sig*lr
		tensor.FMA(relayPointVec, &g, poolVec)
		tensor.FMA(ctxVec, &g, relayPointVec)
	case tensor.Float32:
		var inner float32
		if ip, ok := eng.(tensor.InnerProderF32); ok {
			if inner, err = ip.Inner(ctxVec, relayPointVec); err != nil {
				return
			}
		} else {
			return errors.New("Engine does not perform Inner for Float32")
		}

		sig := model.SigmoidF32(inner)
		g := (float32(1) - float32(childCode) - sig) * float32(lr)
		tensor.FMA(relayPointVec, &g, poolVec)
		tensor.FMA(ctxVec, &g, relayPointVec)

	}
	return nil
}
