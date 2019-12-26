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
	"io"
	"sort"

	"github.com/pkg/errors"
)

// Searcher stores the elements for cosine similarity.
type Searcher struct {
	Vectors   map[string][]float64
	Dimension int
}

// NewSearcher creates *Searcher
func NewSearcher(f io.Reader) (*Searcher, error) {
	vectors := make(map[string][]float64)
	var d int
	storeFunc := func(word string, vec []float64, dim int) {
		if d == 0 {
			d = dim
		} else if d != dim {
			return
		}
		vectors[word] = vec
	}
	if err := ParseAll(f, storeFunc); err != nil {
		return nil, errors.Wrap(err, "Failed to parse")
	}
	return &Searcher{
		Vectors:   vectors,
		Dimension: d,
	}, nil
}

// Search searches similar words for query word and returns top-k nearest neighbors with similarity.
func (s *Searcher) SearchWithQuery(query string, k int) (Neighbors, error) {
	queryVec, ok := s.Vectors[query]
	if !ok {
		return nil, errors.Errorf("%s is not found in vector map", query)
	}
	queryNorm := norm(queryVec)

	if k > len(s.Vectors) {
		k = len(s.Vectors) - 1
	}

	neighbors := make(Neighbors, k)
	for word, vec := range s.Vectors {
		if word == query {
			continue
		}
		n := norm(vec)
		neighbors = append(neighbors, Neighbor{
			word:       word,
			similarity: cosine(queryVec, vec, queryNorm, n),
		})
	}

	sort.Sort(sort.Reverse(neighbors))

	return neighbors[:k], nil

}

func (s *Searcher) Search(queryVec []float64, k int) (Neighbors, error) {
	queryNorm := norm(queryVec)

	if k > len(s.Vectors) {
		k = len(s.Vectors)
	}

	neighbors := make(Neighbors, k)
	for word, vec := range s.Vectors {
		n := norm(vec)
		neighbors = append(neighbors, Neighbor{
			word:       word,
			similarity: cosine(queryVec, vec, queryNorm, n),
		})
	}

	sort.Sort(sort.Reverse(neighbors))

	return neighbors[:k], nil
}
