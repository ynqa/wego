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

package builder

import (
	"testing"
)

func TestInputFile(t *testing.T) {
	b := &Word2VecBuilder{}
	b.InputFile("inputfile")

	if b.inputFile != "inputfile" {
		t.Errorf("Expected builder.inputFile=inputfile: %v", b.inputFile)
	}
}

func TestDimension(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Dimension(100)

	if b.dimension != 100 {
		t.Errorf("Expected builder.dimension=100: %v", b.dimension)
	}
}

func TestIteration(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Iteration(15)

	if b.iteration != 15 {
		t.Errorf("Expected builder.iteration=15: %v", b.iteration)
	}
}

func TestMinCount(t *testing.T) {
	b := &Word2VecBuilder{}
	b.MinCount(5)

	if b.minCount != 5 {
		t.Errorf("Expected builder.minCount=5: %v", b.minCount)
	}
}

func TestThread(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Thread(8)

	if b.thread != 8 {
		t.Errorf("Expected builder.thread=8: %v", b.thread)
	}
}

func TestWindow(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Window(10)

	if b.window != 10 {
		t.Errorf("Expected builder.window=10: %v", b.window)
	}
}

func TestInitLearningRate(t *testing.T) {
	b := &Word2VecBuilder{}
	b.InitLearningRate(0.001)

	if b.initLearningRate != 0.001 {
		t.Errorf("Expected builder.initLearningRate=0.001: %v", b.initLearningRate)
	}
}

func TestToLower(t *testing.T) {
	b := &Word2VecBuilder{}
	b.ToLower()

	if !b.toLower {
		t.Errorf("Expected builder.lower=true: %v", b.toLower)
	}
}

func TestVerbose(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Verbose()

	if !b.verbose {
		t.Errorf("Expected builder.verbose=true: %v", b.verbose)
	}
}

func TestModel(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Model("skip-gram")

	if b.model != "skip-gram" {
		t.Errorf("Expected builder.model=skip-gram: %v", b.model)
	}
}

func TestOptimizer(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Optimizer("ns")

	if b.optimizer != "ns" {
		t.Errorf("Expected builder.optimizer=ns: %v", b.optimizer)
	}
}

func TestMaxDepth(t *testing.T) {
	b := &Word2VecBuilder{}
	b.MaxDepth(40)

	if b.maxDepth != 40 {
		t.Errorf("Expected builder.maxDepth=40: %v", b.maxDepth)
	}
}

func TestNegativeSampleSize(t *testing.T) {
	b := &Word2VecBuilder{}
	b.NegativeSampleSize(20)

	if b.negativeSampleSize != 20 {
		t.Errorf("Expected builder.negativeSampleSize=20: %v", b.negativeSampleSize)
	}
}

func TestTheta(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Theta(1.0e-5)

	if b.theta != 1.0e-5 {
		t.Errorf("Expected builder.theta=1.0e-5: %v", b.theta)
	}
}

func TestBatchSize(t *testing.T) {
	b := &Word2VecBuilder{}
	b.BatchSize(2048)

	if b.batchSize != 2048 {
		t.Errorf("Expected builder.batchSize=2048: %v", b.batchSize)
	}
}

func TestSubSampleThreshold(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SubSampleThreshold(0.001)

	if b.subsampleThreshold != 0.001 {
		t.Errorf("Expected builder.subsampleThreshold=0.001: %v", b.subsampleThreshold)
	}
}

func TestInvalidModelBuild(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Model("fake_model")

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid model except for skip-gram|cbow: %v", b.model)
	}
}

func TestInvalidOptimizerBuild(t *testing.T) {
	b := &Word2VecBuilder{}
	b.Optimizer("fake_optimizer")

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid optimizer except for ns|hs: %v", b.optimizer)
	}
}
