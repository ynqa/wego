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

package search

import (
	"fmt"
	"os"

	"github.com/olekukonko/tablewriter"
	"github.com/pkg/errors"

	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/embedding/embutil"
	"github.com/ynqa/wego/pkg/search/searchutil"
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

type Searcher struct {
	Items embedding.Embeddings
}

func New(embs ...embedding.Embedding) (*Searcher, error) {
	if err := embedding.Embeddings(embs).Validate(); err != nil {
		return nil, err
	}
	return &Searcher{
		Items: embs,
	}, nil
}

func (s *Searcher) SearchInternal(word string, k int) (Neighbors, error) {
	var q embedding.Embedding
	for _, item := range s.Items {
		if item.Word == word {
			q = item
			break
		}
	}
	if q.Word == "" {
		return nil, errors.Errorf("%s is not found in searcher", word)
	}

	neighbors, err := s.Search(q, k, word)
	if err != nil {
		return nil, err
	}
	return neighbors, nil
}

func (s *Searcher) SearchVector(query []float64, k int) (Neighbors, error) {
	return s.Search(embedding.Embedding{
		Vector: query,
		Norm:   embutil.Norm(query),
	}, k)
}

func (s *Searcher) Search(query embedding.Embedding, k int, ignoreWord ...string) (Neighbors, error) {
	neighbors := make(Neighbors, k)

	// Map to quickly check if a word is to be ignored.
	ignoreWords := make(map[string]int, len(ignoreWord))
	for _, word := range ignoreWord {
		ignoreWords[word] = 0
	}

	// Keep track of lowest similarity score.
	low := .0
	for _, item := range s.Items {
		// Drop iteration if the word is to be ignored.
		_, ok := ignoreWords[item.Word]
		if ok {
			continue
		}

		score := searchutil.Cosine(query.Vector, item.Vector, query.Norm, item.Norm)
		// ignore current word if it's similarity is below the lowest score.
		if score > low {
			temp := Neighbor{Word: item.Word, Similarity: score}
			// Bubble up the best match.
			for i := 0; i < len(neighbors); i++ {
				if temp.Similarity > neighbors[i].Similarity {
					temp, neighbors[i] = neighbors[i], temp
					neighbors[i].Rank = uint(i) + 1
				}
			}
			// # neighbors is aways sorted.
			low = neighbors[len(neighbors)-1].Similarity
		}
	}

	// Guard too few items in model.
	for i := 0; i < len(neighbors); i++ {
		if neighbors[i].Word == "" {
			k = i
		}
	}

	return neighbors[:k], nil
}
