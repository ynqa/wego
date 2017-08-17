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
	"testing"

	"github.com/ynqa/word-embedding/utils"
)

func TestHuffmanTree(t *testing.T) {
	fm := utils.FreqMap{
		"a": 1,
		"b": 2,
		"c": 4,
	}

	nm := NewNodeMapFrom(fm)
	nm.BuildHuffmanTree(10)

	if len(nm) != 3 {
		t.Errorf("Expected len=3: %d", len(nm))
	}

	tests := []struct {
		word     string
		expected string
	}{
		{"a", "00"},
		{"b", "01"},
		{"c", "1"},
	}

	for _, test := range tests {
		if nm[test.word].GetPath().Codes() != test.expected {
			t.Errorf("Expected codes: %v, but got %v",
				test.expected, nm[test.word].GetPath().Codes())
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
