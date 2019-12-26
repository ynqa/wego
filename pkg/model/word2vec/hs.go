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
	"github.com/ynqa/wego/pkg/corpus"
	"github.com/ynqa/wego/pkg/node"

	"github.com/pkg/errors"
)

// HierarchicalSoftmax is a piece of Word2Vec optimizer.
type HierarchicalSoftmax struct {
	*SigmoidTable
	nodeMap  map[int]*node.Node
	maxDepth int

	dimension  int
	vocabulary int
}

// NewHierarchicalSoftmax creates *HierarchicalSoftmax.
func NewHierarchicalSoftmax(maxDepth int) *HierarchicalSoftmax {
	return &HierarchicalSoftmax{
		SigmoidTable: newSigmoidTable(),
		maxDepth:     maxDepth,
	}
}

func (hs *HierarchicalSoftmax) initialize(cps *corpus.Word2vecCorpus, dimension int) error {
	nodeMap, err := cps.HuffmanTree(dimension)
	if err != nil {
		return errors.Wrap(err, "Failed to initialize of *HierarchicalSoftmax")
	}
	hs.nodeMap = nodeMap
	hs.dimension = dimension
	hs.vocabulary = cps.Size()
	return nil
}

func (hs *HierarchicalSoftmax) update(word int, lr float64, vector, poolVector []float64) {
	path := hs.nodeMap[word].GetPath()
	for p := 0; p < len(path)-1; p++ {
		relayPoint := path[p]
		childCode := path[p+1].Code
		hs.gradUpd(childCode, lr, relayPoint.Vector, vector, poolVector)
		if hs.maxDepth > 0 && p >= hs.maxDepth {
			break
		}
	}
}

func (hs *HierarchicalSoftmax) gradUpd(childCode int, lr float64, relayPointVec, vector, poolVector []float64) {
	var inner float64
	for i := 0; i < hs.dimension; i++ {
		inner += vector[i] * relayPointVec[i]
	}
	if inner <= -hs.maxExp || inner >= hs.maxExp {
		return
	}
	g := (1.0 - float64(childCode) - hs.sigmoid(inner)) * lr
	for i := 0; i < hs.dimension; i++ {
		poolVector[i] += g * relayPointVec[i]
		relayPointVec[i] += g * vector[i]
	}
}
