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

package opt

import (
	"github.com/ynqa/word-embedding/models"
	"github.com/ynqa/word-embedding/models/word2vec"
	"github.com/ynqa/word-embedding/models/word2vec/huffman"
	"github.com/ynqa/word-embedding/utils"
	"github.com/ynqa/word-embedding/utils/fileio"
	"github.com/ynqa/word-embedding/utils/vector"
)

// HierarchicalSoftmax is a piece of word2vec optimizer.
type HierarchicalSoftmax struct {
	models.Common
	// MaxDepth is the times to dive into huffman tree.
	MaxDepth int
}

// PreTrain executes counting words' frequency, and building huffman tree before training.
func (hs HierarchicalSoftmax) PreTrain() error {
	learningRate = hs.LearningRate
	word2vec.GlobalFreqMap = utils.NewFreqMap()

	if err := fileio.Load(hs.Common.InputFile, word2vec.GlobalFreqMap.Update); err != nil {
		return err
	}

	word2vec.GlobalNodeMap = huffman.NewNodeMapFrom(word2vec.GlobalFreqMap)
	if err := word2vec.GlobalNodeMap.BuildHuffmanTree(hs.Common.Dimension); err != nil {
		return err
	}

	word2vec.GlobalWordMap = word2vec.NewWordMapFrom(word2vec.GlobalFreqMap.Keys(), hs.Common.Dimension, false)
	return nil
}

// Update words' vector using huffman tree.
func (hs HierarchicalSoftmax) Update(target string, contentVector, poolVector vector.Vector) {
	path := word2vec.GlobalNodeMap[target].GetPath()
	for p := 0; p < len(path)-1; p++ {
		relayPoint := path[p]
		f := utils.Sigmoid(contentVector.Inner(relayPoint.Vector))
		childCode := path[p+1].Code
		g := (1.0 - float64(childCode) - f) * learningRate

		for d := 0; d < hs.Common.Dimension; d++ {
			poolVector[d] += g * relayPoint.Vector[d]
			relayPoint.Vector[d] += g * contentVector[d]
		}

		if hs.MaxDepth > 0 && p >= hs.MaxDepth {
			break
		}
	}
}
