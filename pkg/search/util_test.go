// Copyright © 2019 Makoto Ito
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

package search

import (
	"testing"
)

var (
	v1 = []float64{1, 1, 1, 1, 0, 0}
	v2 = []float64{1, 1, 0, 0, 1, 1}
)

func TestNorm(t *testing.T) {
	n1 := norm(v1)
	if n1 != 2. {
		t.Errorf("Expect norm of v1=%v is 2, but got %f", v1, n1)
	}
}

func TestCosine(t *testing.T) {
	n1 := norm(v1)
	n2 := norm(v2)
	sim := cosine(v1, v2, n1, n2)
	if sim != 0.5 {
		t.Errorf("Expect sim(v1, v2)=0.5, but got %f", sim)
	}
}
