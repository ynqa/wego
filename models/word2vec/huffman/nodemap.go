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

package huffman

import (
	"github.com/ynqa/word-embedding/utils"
)

type NodeMap map[string]*Node

func NewNodeMapFrom(f utils.FreqMap) NodeMap {
	nodeMap := make(NodeMap)
	for k, v := range f {
		nodeMap[k] = &Node{
			Value:  v,
			Vector: nil,
		}
	}
	return nodeMap
}

func (n NodeMap) BuildHuffmanTree(nodeVectorDim int) error {
	values := func(nm NodeMap) *Nodes {
		values := make(Nodes, 0)

		for _, v := range nm {
			values = append(values, v)
		}
		return &values
	}

	return values(n).buildHuffmanTree(nodeVectorDim)
}
