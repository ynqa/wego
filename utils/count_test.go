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

package utils

import (
	"github.com/ynqa/word-embedding/utils/set"
	"testing"
)

func TestNewFreqMap(t *testing.T) {
	f := NewFreqMap()

	if len(f) != 0 {
		t.Errorf("Expected len=0: %d", len(f))
	}
}

func TestUpdate(t *testing.T) {
	f := NewFreqMap()
	words := []string{"a", "a", "a", "b", "b", "c"}

	f.Update(words)

	if len(f) != 3 {
		t.Errorf("Expected len=3: %d", len(f))
	}

	if f["a"] != 3 {
		t.Errorf("Expected 'a' freq=3: %d", f["a"])
	}

	if f["b"] != 2 {
		t.Errorf("Expected 'b' freq=2: %d", f["b"])
	}

	if f["c"] != 1 {
		t.Errorf("Expected 'c' freq=1: %d", f["c"])
	}
}

func TestKeys(t *testing.T) {
	f := NewFreqMap()
	words := []string{"a", "a", "a", "b", "b", "c"}

	f.Update(words)

	expected := set.New("a", "b", "c")

	if !f.Keys().Equal(expected) {
		t.Errorf("Expected is %v, but %v", expected, f.Keys())
	}
}
