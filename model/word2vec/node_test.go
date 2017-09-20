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

package word2vec

import (
	"bytes"
	"strconv"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
)

func TestHuffmanTree(t *testing.T) {
	c := testCorpus()
	huffmanTree, err := NewHuffmanTree(c, 100, tensor.Float64, tensor.Float64Engine{})

	if err != nil {
		t.Errorf(err.Error())
	}

	if len(huffmanTree) != 3 {
		t.Errorf("Expected len=3: %d", len(huffmanTree))
	}

	testCases := []struct {
		word     string
		expected string
	}{
		{"A", "00"},
		{"B", "01"},
		{"C", "1"},
	}

	for _, testCase := range testCases {
		wordID, _ := c.Id(testCase.word)
		actual := huffmanTree[wordID].GetPath().Codes()
		if actual != testCase.expected {
			t.Errorf("Expected codes: %v, but got %v in %v",
				testCase.expected, actual, testCase.word)
		}
	}
}

func (n Nodes) Codes() string {
	c := bytes.NewBuffer(make([]byte, 0))
	for _, v := range n {
		c.WriteString(strconv.Itoa(v.Code))
	}
	return c.String()[1:]
}
