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

	"github.com/ynqa/word-embedding/model"
	"github.com/ynqa/word-embedding/model/word2vec/huffman"
	"github.com/ynqa/word-embedding/vector"
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
	hs.nodeMap, err = huffman.NewHuffmanTree(c, dimension)
	return
}

// Update updates the word vector using the huffman tree.
func (hs *HierarchicalSoftmax) Update(targetID int,
	contextVector, poolVector vector.Vector, learningRate float64) error {

	path := hs.nodeMap[targetID].GetPath()
	for p := 0; p < len(path)-1; p++ {
		relayPoint := path[p]

		inner := contextVector.Inner(relayPoint.Vector)
		f := model.Sigmoid(inner)
		childCode := path[p+1].Code
		g := (1.0 - float64(childCode) - f) * learningRate

		poolVector.UnsafeAdd(relayPoint.Vector.Mul(g))
		relayPoint.Vector.UnsafeAdd(contextVector.Mul(g))

		if hs.MaxDepth > 0 && p >= hs.MaxDepth {
			break
		}
	}
	return nil
}
