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
	"testing"

	"github.com/ynqa/word-embedding/corpus"
)

func TestNewHierarchicalSoftmax(t *testing.T) {
	maxDepth := 10
	hs := NewHierarchicalSoftmax(maxDepth)

	if hs.nodeMap != nil {
		t.Error("HierarchicalSoftmax: Initializing without building huffman tree")
	}
}

func TestHSInit(t *testing.T) {
	maxDepth := 10
	hs := NewHierarchicalSoftmax(maxDepth)

	dimension := 10
	hs.initialize(corpus.TestWord2vecCorpus, dimension)

	expectedNodeMapSize := corpus.TestWord2vecCorpus.Size()
	if len(hs.nodeMap) != expectedNodeMapSize {
		t.Errorf("HierarchicalSoftmax: Init returns nodeMap with length=%v: %v",
			expectedNodeMapSize, len(hs.nodeMap))
	}
}
