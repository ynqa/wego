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

package huffman

import (
	"bytes"
	"strconv"
	"strings"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/lingo/corpus"
)

var emptyOpt corpus.ConsOpt = func(c *corpus.Corpus) error { return nil }

func NewDummyCorpus() *corpus.Corpus {
	document := "a b b c c c c"

	c, _ := corpus.Construct(emptyOpt)
	for _, word := range strings.Fields(document) {
		c.Add(word)
	}
	return c
}

func TestHuffmanTree(t *testing.T) {
	c := NewDummyCorpus()
	huffmanTree, err := NewHuffmanTree(c, 100, tensor.Float64, tensor.Float64Engine{})

	if err != nil {
		t.Errorf(err.Error())
	}

	if len(huffmanTree) != 3 {
		t.Errorf("Expected len=3: %d", len(huffmanTree))
	}

	testCases := []struct {
		wordID   int
		expected string
	}{
		{0, "00"},
		{1, "01"},
		{2, "1"},
	}

	for _, testCase := range testCases {
		actual := huffmanTree[testCase.wordID].GetPath().Codes()
		if actual != testCase.expected {
			t.Errorf("Expected codes: %v, but got %v",
				testCase.expected, actual)
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
