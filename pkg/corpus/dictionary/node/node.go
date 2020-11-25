// Copyright © 2020 wego authors
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

package node

type Node struct {
	cache  []*Node
	Parent *Node
	Val    int

	Code   int
	Vector []float64
}

func (n *Node) GetPath(depth int) []*Node {
	if n.cache == nil {
		re := func(nodes []*Node) {
			for i, j := 0, len(nodes)-1; i < j; i, j = i+1, j-1 {
				nodes[i], nodes[j] = nodes[j], nodes[i]
			}
		}
		n.cache = make([]*Node, 0)
		for p := n; p != nil; p = p.Parent {
			n.cache = append(n.cache, p)
		}
		re(n.cache)
	}
	if depth > len(n.cache) {
		depth = len(n.cache)
	}
	return n.cache[:depth]
}
