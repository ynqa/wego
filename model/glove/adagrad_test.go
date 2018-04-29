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

package glove

import (
	"testing"
)

func TestNewAdaGrad(t *testing.T) {
	expectDimension := 10
	expectInitlr := 0.01
	solver := NewAdaGrad(expectDimension, expectInitlr)

	if solver.gradsq != nil {
		t.Error("AdaGrad: gradsq is initialized before calling initialize")
	}

	if solver.dimension != expectDimension {
		t.Errorf("AdaGrad: dimension=%v: %v",
			expectDimension, solver.dimension)
	}

	if solver.initlr != expectInitlr {
		t.Errorf("AdaGrad: initLearningRate=%v: %v",
			expectInitlr, solver.initlr)
	}
}

func TestAdaGradInit(t *testing.T) {
	dimension := 10
	initlr := 0.01
	solver := NewAdaGrad(dimension, initlr)

	expectedVectorSize := 100
	solver.initialize(expectedVectorSize)

	if len(solver.gradsq) != expectedVectorSize {
		t.Errorf("AdaGrad: after init, len(gradsq)=%v: %v", expectedVectorSize, len(solver.gradsq))
	}
}

func TestAdaGradCallBack(t *testing.T) {
	dimension := 10
	initlr := 0.01
	solver := NewAdaGrad(dimension, initlr)

	before := solver.initlr
	solver.postOneIter()
	after := solver.initlr

	if before != after {
		t.Errorf("AdaGrad: without changing after callback: %v -> %v",
			before, after)
	}
}
