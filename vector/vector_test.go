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
		t.Errorf("Expected len=100: %v", len(v))
	}
}

func TestNewRandomizedVector(t *testing.T) {
	v := NewRandomizedVector(100)

	if len(v) != 100 {
		t.Errorf("Expected len=100: %v", len(v))
	}
}

func TestZeros(t *testing.T) {
	v := NewVector(2)

	v[0] = 1
	v[1] = 2

	v.Zeros()
	for _, value := range v {
		if value != 0.0 {
			t.Errorf("Expected zero vector: %v", v)
		}
	}
}

func TestUnsafeAdd(t *testing.T) {
	v1 := NewVector(2)
	v2 := NewVector(2)

	v1[0] = 1
	v1[1] = 2

	v2[0] = 3
	v2[1] = 4

	v1.UnsafeAdd(v2)

	if v1[0] != 4 || v1[1] != 6 {
		t.Errorf("Expected updated v1=[4, 6]: %v", v1)
	}

	if v2[0] != 3 || v2[1] != 4 {
		t.Errorf("Expected immutable v2=[3, 4]: %v", v2)
	}
}

func TestMul(t *testing.T) {
	v := NewVector(2)
	f := 10.0

	v[0] = 1
	v[1] = 2

	res := v.Mul(f)

	if res[0] != 10 || res[1] != 20 {
		t.Errorf("Expected updated res=[10, 20]: %v", res)
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
		t.Errorf("Expected '%v dot %v'=11.0: %v", v1, v2, f)
	}
}

func TestNorm(t *testing.T) {
	v := NewVector(2)

	v[0] = 3
	v[1] = 4

	f := v.Norm()

	if f != 5.0 {
		t.Errorf("Expected %v norm=5.0: %v", v, f)
	}
}

func TestCosine(t *testing.T) {
	v := NewVector(2)

	v[0] = 3
	v[1] = 4

	f := v.Cosine(v)

	if f != 1.0 {
		t.Errorf("Expected cosine=1.0: %v", f)
	}
}

func TestString(t *testing.T) {
	builder := NewVector(10)

	if builder.String() == "" {
		t.Error("String() in Vector shows null")
	}
}
