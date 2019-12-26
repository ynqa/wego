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
	var Fake Config = Config(1024)

	if Fake.String() != "unknown" {
		t.Errorf("Fake should be not registered in Config: %v", Fake.String())
	}
}

func TestConfigString(t *testing.T) {
	testCases := []struct {
		input    Config
		expected string
	}{
		{
			input:    InputFile,
			expected: "inputFile",
		},
		{
			input:    OutputFile,
			expected: "outputFile",
		},
		{
			input:    Dimension,
			expected: "dimension",
		},
		{
			input:    Iteration,
			expected: "iter",
		},
		{
			input:    MinCount,
			expected: "min-count",
		},
		{
			input:    ThreadSize,
			expected: "thread",
		},
		{
			input:    Window,
			expected: "window",
		},
		{
			input:    Initlr,
			expected: "initlr",
		},
		{
			input:    Prof,
			expected: "prof",
		},
		{
			input:    ToLower,
			expected: "lower",
		},
		{
			input:    Verbose,
			expected: "verbose",
		},
		{
			input:    SaveVectorType,
			expected: "save-vec",
		},
		// Word2ec
		{
			input:    Model,
			expected: "model",
		},
		{
			input:    Optimizer,
			expected: "optimizer",
		},
		{
			input:    BatchSize,
			expected: "batchSize",
		},
		{
			input:    MaxDepth,
			expected: "maxDepth",
		},
		{
			input:    NegativeSampleSize,
			expected: "sample",
		},
		{
			input:    SubsampleThreshold,
			expected: "threshold",
		},
		{
			input:    Theta,
			expected: "theta",
		},
		// GloVe
		{
			input:    Solver,
			expected: "solver",
		},
		{
			input:    Xmax,
			expected: "xmax",
		},
		{
			input:    Alpha,
			expected: "alpha",
		},
		// Lexvec
		{
			input:    RelationType,
			expected: "rel",
		},
		// Search
		{
			input:    Rank,
			expected: "rank",
		},
	}

	for _, testCase := range testCases {
		actual := testCase.input.String()
		if actual != testCase.expected {
			t.Errorf("Config: %v with String() should be %v, but get %v", testCase.input, testCase.expected, actual)
		}
	}
}
