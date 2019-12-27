package dictionary

import (
	"sort"

	"github.com/ynqa/wego/pkg/corpus/dictionary/node"
)

func (d *Dictionary) HuffnamTree(dim int) []*node.Node {
	nodes := make([]*node.Node, d.maxid)
	set := make([]*node.Node, d.maxid)
	for i := 0; i < d.maxid; i++ {
		n := &node.Node{
			Val: d.IDFreq(i),
		}
		nodes[i] = n
		set[i] = n
	}

	sort.SliceStable(nodes, func(i, j int) bool {
		return nodes[i].Val < nodes[j].Val
	})
	for len(nodes) > 1 {
		left, right := nodes[0], nodes[1]
		merged := &node.Node{
			Val:    left.Val + right.Val,
			Vector: make([]float64, dim),
		}
		left.Code, right.Code = 0, 1
		left.Parent, right.Parent = merged, merged

		nodes = nodes[2:]
		idx := sort.Search(len(nodes), func(i int) bool {
			return nodes[i].Val >= merged.Val
		})

		nodes = append(nodes, &node.Node{})
		copy(nodes[idx+1:], nodes[idx:])
		nodes[idx] = merged
	}

	return set
}
