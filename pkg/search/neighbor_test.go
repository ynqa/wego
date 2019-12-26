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

package search

import (
	"testing"
)

func NewFakeNeighbors() Neighbors {
	ns := make(Neighbors, 0)
	ns = append(ns, Neighbor{
		word:       "Cupcake",
		similarity: 0.1,
	})
	ns = append(ns, Neighbor{
		word:       "Donut",
		similarity: 0.2,
	})
	ns = append(ns, Neighbor{
		word:       "Eclair",
		similarity: 0.3,
	})
	ns = append(ns, Neighbor{
		word:       "Froyo",
		similarity: 0.4,
	})
	ns = append(ns, Neighbor{
		word:       "Gingerbread",
		similarity: 0.5,
	})
	return ns
}

func TestNeighborsLen(t *testing.T) {
	neighbors := NewFakeNeighbors()

	if neighbors.Len() != 5 {
		t.Errorf("Expected len=5: %v", neighbors.Len())
	}
}

func TestNeighborsLess(t *testing.T) {
	neighbors := NewFakeNeighbors()

	if !neighbors.Less(0, 3) {
		t.Errorf("Expected less(0, 3)=true: neighbors[0].similarity %v vs. neighbors[3].similarity %v",
			neighbors[0].similarity, neighbors[3].similarity)
	}
}

func TestNeighborsSwap(t *testing.T) {
	neighbors := NewFakeNeighbors()
	n0 := neighbors[0]
	n3 := neighbors[3]

	neighbors.Swap(0, 3)

	if neighbors[0] != n3 {
		t.Errorf("Expected to equal %v to %v", neighbors[0], n3)
	}

	if neighbors[3] != n0 {
		t.Errorf("Expected to equal %v to %v", neighbors[3], n0)
	}
}
