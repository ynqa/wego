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

func TestInvalidConfigString(t *testing.T) {
	var Fake Config = 1024

	if Fake.String() != "unknown" {
		t.Errorf("Fake should be not registered in Config: %v", Fake.String())
	}
}

func TestConfigString(t *testing.T) {
	testCases := []struct {
		input  Config
		expect string
	}{
		{
			input:  InputFile,
			expect: "inputFile",
		},
		{
			input:  OutputFile,
			expect: "outputFile",
		},
		{
			input:  Dimension,
			expect: "dimension",
		},
		{
			input:  Window,
			expect: "window",
		},
		{
			input:  InitLearningRate,
			expect: "initlr",
		},
		{
			input:  Prof,
			expect: "prof",
		},
		{
			input:  ToLower,
			expect: "lower",
		},
		{
			input:  Verbose,
			expect: "verbose",
		},
	}

	for _, testCase := range testCases {
		actual := testCase.input.String()
		if actual != testCase.expect {
			t.Errorf("Config: %v with String() should be %v, but get %v", testCase.input, testCase.expect, actual)
		}
	}
}
