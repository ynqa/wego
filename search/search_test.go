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

var testVector = `apple 1 1 1 1 1
	banana 1 1 1 1 1
	chocolate 0 0 0 0 0
	dragon -1 -1 -1 -1 -1`

func TestSearch(t *testing.T) {
	searcher := NewSearcher("apple", 3)

	f := ioutil.NopCloser(bytes.NewReader([]byte(testVector)))
	err := searcher.Search(f)

	if err != nil {
		t.Errorf(err.Error())
	}

	if len(searcher.dense) != 4 {
		t.Errorf("Expected estimator.tensor len=4: %d", len(searcher.dense))
	}
}
