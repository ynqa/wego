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
