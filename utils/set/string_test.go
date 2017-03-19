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

package set

import (
	"testing"
)

func TestNew(t *testing.T) {
	s := New()

	if len(s) != 0 {
		t.Errorf("Expected len=0: %d", len(s))
	}

	s1 := New("a", "b", "c")

	if len(s1) != 3 {
		t.Errorf("Expected len=3: %d", len(s))
	}
}

func TestAdd(t *testing.T) {
	s := New()

	s.Add("a", "b", "c")

	if len(s) != 3 {
		t.Errorf("Expected len=3: %d", len(s))
	}
}

func TestContain(t *testing.T) {
	s := New("a", "b", "c")

	if !s.Contain("a") || !s.Contain("b") || !s.Contain("c") {
		t.Errorf("Missing contents 'a', 'b', 'c' in: %v", s)
	}

	if s.Contain("d") {
		t.Errorf("Unexpected contents 'd' in: %v", s)
	}
}

func TestEqual(t *testing.T) {
	s := New("a", "b", "c")

	s1 := New("a", "b")

	if !s.Equal(s) {
		t.Errorf("Expected to be equal: %v = %v", s, s)
	}

	if s.Equal(s1) {
		t.Errorf("Expected to be not equal: %v = %v", s, s1)
	}
}
