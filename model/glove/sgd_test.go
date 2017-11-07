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

func TestNewSGD(t *testing.T) {
	solver := NewSGD(conf)

	if solver.dimension != config.DefaultDimension {
		t.Errorf("SGD: dimension=%v: %v",
			config.DefaultDimension, solver.dimension)
	}

	if solver.currentLearningRate != config.DefaultInitLearningRate {
		t.Errorf("SGD: currentLearningRate=%v: %v",
			config.DefaultInitLearningRate, solver.currentLearningRate)
	}
}

func TestSGDCallBack(t *testing.T) {
	solver := NewSGD(conf)

	before := solver.currentLearningRate
	solver.callback()
	after := solver.currentLearningRate

	if before < after {
		t.Errorf("SGD: currentLearningRate is smaller than after callback: %v -> %v",
			before, after)
	}
}
