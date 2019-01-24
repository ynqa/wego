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
	"bytes"
	"io/ioutil"
	"testing"
)

func TestSearchWithQuery(t *testing.T) {
	f := ioutil.NopCloser(bytes.NewReader([]byte(testVectorStr)))
	defer f.Close()

	searcher, err := NewSearcher(f)
	if err != nil {
		t.Errorf("Failed to create searcher: %s", err.Error())
	}

	neighbors, err := searcher.SearchWithQuery("banana", 20)
	if err != nil {
		t.Errorf("Failed to search with word=banana, rank=20: %s", err.Error())
	}

	if len(searcher.Vectors) != testNumVector {
		t.Errorf("Expected searcher.Vectors len=%d, but got %d", testNumVector, len(searcher.Vectors))
	}

	if len(neighbors) != testNumVector-1 {
		t.Errorf("Expected neighbors len=%d, but got %d", testNumVector-1, len(neighbors))
	}

	if neighbors[0].word != "apple" {
		t.Errorf("Expected the most near word is `apple` for `banana`, but got neighbors=%v", neighbors)
	}
}

func TestSearch(t *testing.T) {
	f := ioutil.NopCloser(bytes.NewReader([]byte(testVectorStr)))
	defer f.Close()

	searcher, err := NewSearcher(f)
	if err != nil {
		t.Errorf("Failed to create searcher: %s", err.Error())
	}

	neighbors, err := searcher.Search(dragonVector, 20)
	if err != nil {
		t.Errorf("Failed to search with vector=%v, rank=20: %s", dragonVector, err.Error())
	}

	if len(searcher.Vectors) != testNumVector {
		t.Errorf("Expected searcher.Vectors len=%d, but got %d", testNumVector, len(searcher.Vectors))
	}

	if len(neighbors) != testNumVector {
		t.Errorf("Expected neighbors len=%d, but got %d", testNumVector, len(neighbors))
	}

	if neighbors[0].word != "dragon" {
		t.Errorf("Expected the most near word is vector=%v for `dragon`, but got neighbors=%v", dragonVector, neighbors)
	}
}
