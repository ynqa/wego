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

func TestSearch(t *testing.T) {
	f := ioutil.NopCloser(bytes.NewReader([]byte(testVector)))
	defer f.Close()
	parser := NewParser(f)

	searcher, err := NewSearcher(parser)
	if err != nil {
		t.Errorf("Failed to create searcher: %s", err.Error())
	}

	neighbors, err := searcher.Search("banana", 20)
	if err != nil {
		t.Errorf("Failed to search with word=banana, rank=20: %s", err.Error())
	}

	if len(searcher.vectors) != testNumVector {
		t.Errorf("Expected searcher.vectors len=%d, but got %d", testNumVector, len(searcher.vectors))
	}

	if len(neighbors) != 3 {
		t.Errorf("Expected neighbors len=3, but got %d", len(neighbors))
	}

	if neighbors[0].word != "apple" {
		t.Errorf("Expected the most near word is apple for banana, but got %s", neighbors[0].word)
	}
}
