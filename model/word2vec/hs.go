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
)

// HierarchicalSoftmax is a piece of Word2Vec optimizer.
type HierarchicalSoftmax struct {
	nodeMap  map[int]*Node
	maxDepth int

	vocabulary int
	dimension  int
}

// NewHierarchicalSoftmax creates *HierarchicalSoftmax.
// The huffman tree is NOT built yet.
func NewHierarchicalSoftmax(maxDepth int) *HierarchicalSoftmax {
	hs := new(HierarchicalSoftmax)
	hs.maxDepth = maxDepth
	return hs
}

// Init initializes the huffman tree.
func (hs *HierarchicalSoftmax) Init(c *corpus.Corpus, dimension int) (err error) {
	hs.vocabulary = c.Size()
	hs.dimension = dimension

	hs.nodeMap, err = NewHuffmanTree(c, dimension)
	return
}

// Update updates the word vector using the huffman tree.
func (hs *HierarchicalSoftmax) Update(targetID int, contextVector, poolVector []float64, learningRate float64) {
	path := hs.nodeMap[targetID].GetPath()
	for p := 0; p < len(path)-1; p++ {
		relayPoint := path[p]

		childCode := path[p+1].Code

		hs.gradUpd(childCode, learningRate, relayPoint.Vector, poolVector, contextVector)

		if hs.maxDepth > 0 && p >= hs.maxDepth {
			break
		}
	}
}

func (hs *HierarchicalSoftmax) gradUpd(childCode int, lr float64, relayPointVec, poolVec, ctxVec []float64) {
	var inner float64

	for i := 0; i < hs.dimension; i++ {
		inner += ctxVec[i] * relayPointVec[i]
	}
	g := (1.0 - float64(childCode) - sigmoid(inner)) * lr

	for i := 0; i < hs.dimension; i++ {
		poolVec[i] += g * relayPointVec[i]
		relayPointVec[i] += g * ctxVec[i]
	}
}
