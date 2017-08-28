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

package vector

import (
	"testing"
)

func TestNewVector(t *testing.T) {
	v := NewVector(100)

	if len(v) != 100 {
		t.Errorf("Expected len=3: %d", len(v))
	}
}

func TestInner(t *testing.T) {
	v1 := NewVector(2)
	v2 := NewVector(2)

	v1[0] = 1
	v1[1] = 2

	v2[0] = 3
	v2[1] = 4

	f := v1.Inner(v2)

	if f != 11.0 {
		t.Errorf("Expected '%v dot %v'=11.0: %f", v1, v2, f)
	}
}

func TestNorm(t *testing.T) {
	v := NewVector(2)

	v[0] = 3
	v[1] = 4

	f := v.Norm()

	if f != 5.0 {
		t.Errorf("Expected %v norm=5.0: %f", v, f)
	}
}

func TestCosine(t *testing.T) {
	v := NewVector(2)

	v[0] = 3
	v[1] = 4

	f := v.Cosine(v)

	if f != 1.0 {
		t.Errorf("Expected cosine=1.0: %f", f)
	}
}
