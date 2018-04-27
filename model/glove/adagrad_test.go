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

	"github.com/ynqa/word-embedding/config"
)

func TestNewAdaGrad(t *testing.T) {
	solver := NewAdaGrad(conf)

	if solver.gradsq != nil {
		t.Error("AdaGrad: Initializing gradsq before Preprocess")
	}

	if solver.dimension != config.DefaultDimension {
		t.Errorf("AdaGrad: dimension=%v: %v",
			config.DefaultDimension, solver.dimension)
	}

	if solver.initLearningRate != config.DefaultInitLearningRate {
		t.Errorf("AdaGrad: initLearningRate=%v: %v",
			config.DefaultInitLearningRate, solver.initLearningRate)
	}
}

func TestAdaGradInit(t *testing.T) {
	solver := NewAdaGrad(conf)

	solver.initialize(100)

	if len(solver.gradsq) != 100 {
		t.Errorf("AdaGrad: after init, len(gradsq)=100: %v", len(solver.gradsq))
	}
}

func TestAdaGradCallBack(t *testing.T) {
	solver := NewAdaGrad(conf)

	before := solver.initLearningRate
	solver.postOneIter()
	after := solver.initLearningRate

	if before != after {
		t.Errorf("AdaGrad: without changing after callback: %v -> %v",
			before, after)
	}
}
