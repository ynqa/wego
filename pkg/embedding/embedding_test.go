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

package embedding

import (
	"bytes"
	"io/ioutil"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/ynqa/wego/pkg/embedding/embutil"
)

func TestLoad(t *testing.T) {
	testCases := []struct {
		name     string
		contents string
		itemSize int
	}{
		{
			name: "read vector file",
			contents: `apple 1 1 1 1 1
			banana 1 1 1 1 1
			chocolate 0 0 0 0 0
			dragon -1 -1 -1 -1 -1`,
			itemSize: 4,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			embs, _ := Load(bytes.NewReader([]byte(tc.contents)))
			assert.Equal(t, tc.itemSize, len(embs))
		})
	}
}

func TestParse(t *testing.T) {
	testNumVector := 4
	testVectorStr := `apple 1 1 1 1 1
banana 1 1 1 1 1
chocolate 0 0 0 0 0
dragon -1 -1 -1 -1 -1`

	f := ioutil.NopCloser(bytes.NewReader([]byte(testVectorStr)))
	defer f.Close()

	embs := make([]Embedding, 0)
	op := func(emb Embedding) error {
		embs = append(embs, emb)
		return nil
	}

	assert.NoError(t, parse(f, op))
	assert.Equal(t, testNumVector, len(embs))
}

func TestParseLine(t *testing.T) {
	testCases := []struct {
		name     string
		line     string
		expected Embedding
	}{
		{
			name: "parse line into Embedding",
			line: "apple 1 1 1 1 1",
			expected: Embedding{
				Word:   "apple",
				Dim:    5,
				Vector: []float64{1, 1, 1, 1, 1},
				Norm:   embutil.Norm([]float64{1, 1, 1, 1, 1}),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			emb, _ := parseLine(tc.line)
			assert.Truef(t, reflect.DeepEqual(tc.expected, emb), "Must be equal %v and %v", tc.expected, emb)
		})
	}
}
