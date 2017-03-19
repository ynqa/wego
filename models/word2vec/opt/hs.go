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

type HierarchicalSoftmax struct {
	models.Common
	MaxDepth int
}

func (hs HierarchicalSoftmax) PreTrain() error {
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

func (hs HierarchicalSoftmax) Update(target string, contentOrSumVector, poolVector vector.Vector) {
	path := word2vec.GlobalNodeMap[target].GetPath()
	for p := 0; p < len(path)-1; p++ {
		relayPoint := path[p]
		f := utils.Sigmoid(contentOrSumVector.Inner(relayPoint.Vector))
		childCode := path[p+1].Code
		g := (1. - float64(childCode) - f) * hs.Common.LearningRate

		for d := 0; d < hs.Common.Dimension; d++ {
			poolVector[d] += g * relayPoint.Vector[d]
			relayPoint.Vector[d] += g * contentOrSumVector[d]
		}

		if hs.MaxDepth > 0 && p >= hs.MaxDepth {
			break
		}
	}
}
