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
	"sort"

	"github.com/pkg/errors"
)

// Searcher stores the elements for cosine similarity.
type Searcher struct {
	vectors map[string][]float64
}

// NewSearcher creates *Searcher
func NewSearcher(parser *Parser) (*Searcher, error) {
	vectors := make(map[string][]float64)
	storeFunc := func(word string, vec []float64) {
		vectors[word] = vec
	}
	if err := parser.ParseAll(storeFunc); err != nil {
		return nil, errors.Wrap(err, "Failed to parse")
	}
	return &Searcher{
		vectors: vectors,
	}, nil
}

// Search searches similar words for query word and returns top-k nearest neighbors with similarity.
func (s *Searcher) Search(query string, k int) (Neighbors, error) {
	queryVec, ok := s.vectors[query]
	if !ok {
		return nil, errors.Errorf("%s is not found in vector map", query)
	}
	queryNorm := norm(queryVec)

	if k > len(s.vectors) {
		k = len(s.vectors) - 1
	}
	neighbors := make(Neighbors, k)
	for word, vec := range s.vectors {
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
