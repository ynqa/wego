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
	"math"
	"testing"
)

func TestNextRandom(t *testing.T) {
	// TODO: Fuzzy Test
	r := nextRandom(5)
	if !(0 <= r && r < 5) {
		t.Errorf("Extected range between 0 < nextRandom(x) < 5: %v", r)
	}
}

func TestSigmoid(t *testing.T) {
	f := sigmoid(0)
	if f != 0.5 {
		t.Errorf("Expected SigmoidF64(0.0)=0.5: %v", f)
	}
}

func TestPositiveInfSigmoid(t *testing.T) {
	f := sigmoid(math.Inf(1))

	if f != 1.0 {
		t.Errorf("Expected SigmoidF64(+Inf)=1.0: %v", f)
	}
}

func TestNegativeInfSigmoid(t *testing.T) {
	f := sigmoid(math.Inf(-1))

	if f != 0.0 {
		t.Errorf("Expected SigmoidF64(-Inf)=0.0: %v", f)
	}
}
