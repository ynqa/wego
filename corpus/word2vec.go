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

package corpus

import (
	"io"

	"github.com/ynqa/word-embedding/corpus/node"
)

// Word2VecCorpus stores corpus itself by integer word id list.
type Word2VecCorpus struct {
	*core
}

// NewWord2VecCorpus creates *Word2VecCorpus.
func NewWord2VecCorpus(f io.ReadCloser, toLower bool, minCount int) *Word2VecCorpus {
	word2VecCorpus := &Word2VecCorpus{
		core: newCore(),
	}
	word2VecCorpus.parse(f, toLower, minCount)
	return word2VecCorpus
}

// HuffmanTree builds word nodes map.
func (wc *Word2VecCorpus) HuffmanTree(dimension int) (map[int]*node.Node, error) {
	ns := make(node.Nodes, 0, wc.Size())
	nm := make(map[int]*node.Node)
	for i := 0; i < wc.Size(); i++ {
		n := new(node.Node)
		n.Value = wc.IDFreq(i)
		nm[i] = n
		ns = append(ns, n)
	}
	if err := ns.Build(dimension); err != nil {
		return nil, err
	}
	return nm, nil
}
