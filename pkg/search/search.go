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
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/olekukonko/tablewriter"
	"github.com/pkg/errors"

	"github.com/ynqa/wego/pkg/item"
)

// Neighbor stores the word with cosine similarity value on the target.
type Neighbor struct {
	Word       string
	Rank       uint
	Similarity float64
}

type Neighbors []Neighbor

func (neighbors Neighbors) Describe() {
	table := make([][]string, len(neighbors))
	for i, n := range neighbors {
		table[i] = []string{
			fmt.Sprintf("%d", n.Rank),
			n.Word,
			fmt.Sprintf("%f", n.Similarity),
		}
	}

	writer := tablewriter.NewWriter(os.Stdout)
	writer.SetHeader([]string{"Rank", "Word", "Similarity"})
	writer.SetBorder(false)
	writer.AppendBulk(table)
	writer.Render()
}

type Item struct {
	item.Item
	Norm float64
}

type Items []Item

func (items Items) Empty() bool {
	return len(items) == 0
}

func (items Items) Find(word string) (Item, bool) {
	for _, item := range items {
		if word == item.Word {
			return item, true
		}
	}
	return Item{}, false
}

type Searcher struct {
	Items Items
}

func New(items ...item.Item) (*Searcher, error) {
	elems := make(Items, len(items))
	wholeDim := 0
	for i, item := range items {
		if err := item.Validate(); err != nil {
			return nil, err
		}
		if i != 0 && wholeDim != item.Dim {
			return nil, errors.Errorf("whole of word Dim for searcher must be same, maybe %d but got %d", wholeDim, item.Dim)
		}
		elems[i] = Item{
			Item: item,
			Norm: norm(item.Vector),
		}
		wholeDim = item.Dim
	}
	return &Searcher{
		Items: elems,
	}, nil
}

func NewForVectorFile(r io.Reader) (*Searcher, error) {
	var (
		elems    Items
		i        int
		wholeDim int
	)
	err := item.Parse(r, item.ItemOp(func(item item.Item) error {
		if err := item.Validate(); err != nil {
			return err
		}
		if i != 0 && wholeDim != item.Dim {
			return errors.Errorf("whole of dim for searcher must be same, maybe %d but got %d", wholeDim, item.Dim)
		}
		elems = append(elems, Item{
			Item: item,
			Norm: norm(item.Vector),
		})
		i++
		wholeDim = item.Dim
		return nil
	}))
	if err != nil {
		return nil, err
	}
	return &Searcher{
		Items: elems,
	}, nil
}

func (s *Searcher) InternalSearch(word string, k int) (Neighbors, error) {
	var q Item
	for _, item := range s.Items {
		if item.Word == word {
			q = item
			break
		}
	}
	if q.Word == "" {
		return nil, errors.Errorf("%s is not found in searcher", word)
	}

	neighbors, err := s.search(q, k, word)
	if err != nil {
		return nil, err
	}

	idx := -1
	for i, neighbor := range neighbors {
		if neighbor.Word == word {
			idx = i
			break
		}
	}

	if idx >= 0 {
		neighbors = append(neighbors[:idx], neighbors[idx+1:]...)
	}
	return neighbors, nil
}

func (s *Searcher) Search(query []float64, k int) (Neighbors, error) {
	return s.search(Item{
		Item: item.Item{
			Vector: query,
		},
		Norm: norm(query),
	}, k)
}

func (s *Searcher) search(query Item, k int, ignoreWord ...string) (Neighbors, error) {
	neighbors := make(Neighbors, len(s.Items))
	for i, item := range s.Items {
		var ignore bool
		for _, w := range ignoreWord {
			ignore = ignore || item.Word == w
		}
		if !ignore {
			neighbors[i] = Neighbor{
				Word:       item.Word,
				Similarity: cosine(query.Vector, item.Vector, query.Norm, item.Norm),
			}
		}
	}

	sort.SliceStable(neighbors, func(i, j int) bool {
		return neighbors[i].Similarity > neighbors[j].Similarity
	})
	for i := range neighbors {
		neighbors[i].Rank = uint(i) + 1
	}
	if k > len(s.Items) {
		k = len(s.Items)
	}
	return neighbors[:k], nil
}

func norm(vec []float64) float64 {
	var n float64
	for _, v := range vec {
		n += math.Pow(v, 2)
	}
	return math.Sqrt(n)
}

func cosine(v1, v2 []float64, n1, n2 float64) float64 {
	if n1 == 0 || n2 == 0 {
		return 0
	}
	var dot float64
	for i := range v1 {
		dot += v1[i] * v2[i]
	}
	return dot / n1 / n2
}
