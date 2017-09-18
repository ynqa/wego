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

package config

import (
	"testing"
)

func TestInvalidWord2VecConfigString(t *testing.T) {
	var Fake Word2VecConfig = 1024

	if Fake.String() != "unknown" {
		t.Errorf("Fake should be not registered in Word2VecConfig: %v", Fake.String())
	}
}

func TestWord2VecConfigString(t *testing.T) {
	testCases := []struct {
		input  Word2VecConfig
		expect string
	}{
		{
			input:  Model,
			expect: "model",
		},
		{
			input:  Optimizer,
			expect: "optimizer",
		},
		{
			input:  MaxDepth,
			expect: "maxDepth",
		},
		{
			input:  NegativeSampleSize,
			expect: "sample",
		},
		{
			input:  Theta,
			expect: "theta",
		},
		{
			input:  BatchSize,
			expect: "batchSize",
		},
		{
			input:  SubsampleThreshold,
			expect: "threshold",
		},
	}

	for _, testCase := range testCases {
		actual := testCase.input.String()
		if actual != testCase.expect {
			t.Errorf("Word2VecConfig: %v with String() should be %v, but get %v", testCase.input, testCase.expect, actual)
		}
	}
}
