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
)

func TestNextRandom(t *testing.T) {
	// TODO: Fuzzy Test
	r := nextRandom(5)
	if !(0 <= r && r < 5) {
		t.Errorf("Extected range between 0 < nextRandom(x) < 5: %v", r)
	}
}

// Prove that the learning rate is decreasing as currentWords is higher,
func TestComputeLearningRate(t *testing.T) {
	initlr := 0.025
	theta := 1.0e-4
	totalWords := 100
	firstCurrentWords := 50

	flr := computeLearningRate(initlr, theta, totalWords, firstCurrentWords)

	secondCurrentWords := 70

	slr := computeLearningRate(initlr, theta, totalWords, secondCurrentWords)

	if flr < slr {
		t.Errorf("Expected first > second: %v vs. %v", flr, slr)
	}
}

// Prove that the learning rate is not less than initlr * theta
func TestLowerLimitLearningRate(t *testing.T) {
	initlr := 0.025
	theta := 1.0e-4
	currentWords := 100
	totalWords := 100

	lr := computeLearningRate(initlr, theta, currentWords, totalWords)

	if lr != (initlr * theta) {
		t.Errorf("Expected that the lower limit of learning rate is (initlr * theta)=%v: %v",
			(initlr * theta), lr)
	}
}
