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
	"math/rand"
	"sort"

	"github.com/chewxy/lingo/corpus"
)

// Node stores the node with vector in huffman tree.
type Node struct {
	Parent    *Node
	Code      int
	Value     int
	Vector    []float64
	CachePath Nodes
}

// Nodes is the list of Node.
type Nodes []*Node

func (n *Nodes) Len() int           { return len(*n) }
func (n *Nodes) Less(i, j int) bool { return (*n)[i].Value < (*n)[j].Value }
func (n *Nodes) Swap(i, j int)      { (*n)[i], (*n)[j] = (*n)[j], (*n)[i] }

// NewHuffmanTree creates the map of wordID with Node.
func NewHuffmanTree(c *corpus.Corpus, dimension int) (map[int]*Node, error) {
	ns := make(Nodes, 0, c.Size())
	nm := make(map[int]*Node)
	for i := 0; i < c.Size(); i++ {
		n := new(Node)
		n.Value = c.IDFreq(i)
		nm[i] = n
		ns = append(ns, n)
	}
	err := ns.buildHuffmanTree(dimension)
	if err != nil {
		return nil, err
	}
	return nm, nil
}

func (n *Nodes) buildHuffmanTree(dimension int) error {
	if len(*n) == 0 {
		return errors.New("The length of Nodes is 0")
	}
	sort.Sort(n)

	for len(*n) > 1 {
		// Pop
		left, right := (*n)[0], (*n)[1]
		*n = (*n)[2:]

		vec := make([]float64, dimension)
		for i := 0; i < dimension; i++ {
			vec[i] = (rand.Float64() - 0.5) / float64(dimension)
		}

		parentValue := left.Value + right.Value
		parent := &Node{
			Value:  parentValue,
			Vector: vec,
		}
		left.Parent = parent
		left.Code = 0
		right.Parent = parent
		right.Code = 1

		idx := sort.Search(len(*n), func(i int) bool { return (*n)[i].Value >= parentValue })

		// Insert
		*n = append(*n, &Node{})
		copy((*n)[idx+1:], (*n)[idx:])
		(*n)[idx] = parent
	}
	return nil
}

// GetPath returns the nodes from root to word on huffman tree.
func (n *Node) GetPath() Nodes {
	// Reverse
	re := func(n Nodes) {
		for i, j := 0, len(n)-1; i < j; i, j = i+1, j-1 {
			n[i], n[j] = n[j], n[i]
		}
	}

	trace := func() Nodes {
		nodes := make(Nodes, 0)
		nodes = append(nodes, n)
		for parent := n.Parent; parent != nil; parent = parent.Parent {
			nodes = append(nodes, parent)
		}
		re(nodes)
		return nodes
	}

	path := n.CachePath
	if path == nil {
		path = trace()
	}

	n.CachePath = path
	return path
}
