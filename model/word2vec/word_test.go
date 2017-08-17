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
	"testing"

	"github.com/ynqa/word-embedding/utils/set"
)

func TestNewWordMapFrom(t *testing.T) {
	s := set.New("a", "b", "c")

	wm1 := NewWordMapFrom(s, 10, false)

	if len(wm1) != 3 {
		t.Errorf("Expected len=3: %d", len(wm1))
	}

	if len(wm1["a"].Vector) != 10 {
		t.Errorf("Expected len=10: %d", len(wm1["a"].Vector))
	}

	if wm1["a"].VectorAsNegative != nil {
		t.Error("Expected nil")
	}

	wm2 := NewWordMapFrom(s, 10, true)

	if len(wm2["a"].VectorAsNegative) != 10 {
		t.Errorf("Expected len=10: %d", len(wm2["a"].VectorAsNegative))
	}
}
