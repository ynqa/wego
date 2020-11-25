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

package cpsutil

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReadWord(t *testing.T) {
	var dic []string
	fn := func(w string) (err error) {
		dic = append(dic, w)
		return
	}

	r := strings.NewReader("a bc def")
	expected := []string{"a", "bc", "def"}
	assert.NoError(t, ReadWord(r, fn))
	assert.Equal(t, expected, dic)
}

func TestReadWordWithForwardContext(t *testing.T) {
	var dic []string
	fn := func(w1, w2 string) (err error) {
		dic = append(dic, w1+w2)
		return
	}

	r := strings.NewReader("a b c d e")
	expected := []string{"ab", "ac", "bc", "bd", "cd", "ce", "de"}
	assert.NoError(t, ReadWordWithForwardContext(r, 2, fn))
	assert.Equal(t, expected, dic)
}
