// Copyright © 2019 Makoto Ito
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

package item

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"reflect"
	"testing"
)

func TestParse(t *testing.T) {
	testNumVector := 4
	testVectorStr := `apple 1 1 1 1 1
banana 1 1 1 1 1
chocolate 0 0 0 0 0
dragon -1 -1 -1 -1 -1`

	f := ioutil.NopCloser(bytes.NewReader([]byte(testVectorStr)))
	defer f.Close()

	items := make([]Item, 0)
	op := func(item Item) error {
		items = append(items, item)
		return nil
	}

	assert.NoError(t, Parse(f, op))
	assert.Equal(t, testNumVector, len(items))
}

func TestParseLine(t *testing.T) {
	testCases := []struct {
		name     string
		line     string
		expected Item
	}{
		{
			name: "parse line into Item",
			line: "apple 1 1 1 1 1",
			expected: Item{
				Word:   "apple",
				Dim:    5,
				Vector: []float64{1, 1, 1, 1, 1},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			item, _ := ParseLine(tc.line)
			assert.Truef(t, reflect.DeepEqual(tc.expected, item), "Must be equal %v and %v", tc.expected, item)
		})
	}
}
