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
	"github.com/ynqa/wego/pkg/model/word2vec"
	"testing"
)

func TestWord2vecDimension(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedDimension := 100
	b.Dimension(expectedDimension)

	if b.dimension != expectedDimension {
		t.Errorf("Expected builder.dimension=%v: %v", expectedDimension, b.dimension)
	}
}

func TestWord2vecIteration(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedIteration := 50
	b.Iteration(expectedIteration)

	if b.iteration != expectedIteration {
		t.Errorf("Expected builder.iteration=%v: %v", expectedIteration, b.iteration)
	}
}

func TestWord2vecMinCount(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedMinCount := 10
	b.MinCount(expectedMinCount)

	if b.minCount != expectedMinCount {
		t.Errorf("Expected builder.minCount=%v: %v", expectedMinCount, b.minCount)
	}
}

func TestWord2vecThreadSize(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedThreadSize := 8
	b.ThreadSize(expectedThreadSize)

	if b.threadSize != expectedThreadSize {
		t.Errorf("Expected builder.threadSize=%v: %v", expectedThreadSize, b.threadSize)
	}
}

func TestWord2vecWindow(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedWindow := 10
	b.Window(expectedWindow)

	if b.window != expectedWindow {
		t.Errorf("Expected builder.window=%v: %v", expectedWindow, b.window)
	}
}

func TestWord2vecInitlr(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedInitlr := 0.001
	b.Initlr(expectedInitlr)

	if b.initlr != expectedInitlr {
		t.Errorf("Expected builder.initlr=%v: %v", expectedInitlr, b.initlr)
	}
}

func TestWord2vecToLower(t *testing.T) {
	b := &Word2vecBuilder{}

	b.ToLower()

	if !b.toLower {
		t.Errorf("Expected builder.lower=true: %v", b.toLower)
	}
}

func TestWord2vecVerbose(t *testing.T) {
	b := &Word2vecBuilder{}

	b.Verbose()

	if !b.verbose {
		t.Errorf("Expected builder.verbose=true: %v", b.verbose)
	}
}

func TestWord2vecModel(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedModel := word2vec.CBOW
	b.Model(expectedModel)

	if b.model != expectedModel {
		t.Errorf("Expected builder.model=%v: %v", expectedModel, b.model)
	}
}

func TestWord2vecOptimizer(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedOptimizer := word2vec.NEGATIVE_SAMPLING
	b.Optimizer(expectedOptimizer)

	if b.optimizer != expectedOptimizer {
		t.Errorf("Expected builder.optimizer=%v: %v", expectedOptimizer, b.optimizer)
	}
}

func TestWord2vecBatchSize(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedBatchSize := 2048
	b.BatchSize(expectedBatchSize)

	if b.batchSize != expectedBatchSize {
		t.Errorf("Expected builder.batchSize=%v: %v", expectedBatchSize, b.batchSize)
	}
}

func TestWord2vecMaxDepth(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedMaxDepth := 40
	b.MaxDepth(expectedMaxDepth)

	if b.maxDepth != expectedMaxDepth {
		t.Errorf("Expected builder.maxDepth=%v: %v", expectedMaxDepth, b.maxDepth)
	}
}

func TestWord2vecNegativeSampleSize(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedNegativeSampleSize := 20
	b.NegativeSampleSize(expectedNegativeSampleSize)

	if b.negativeSampleSize != expectedNegativeSampleSize {
		t.Errorf("Expected builder.negativeSampleSize=%v: %v", expectedNegativeSampleSize, b.negativeSampleSize)
	}
}

func TestWord2vecSubSampleThreshold(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedSubSampleThreshold := 0.001
	b.SubSampleThreshold(expectedSubSampleThreshold)

	if b.subsampleThreshold != expectedSubSampleThreshold {
		t.Errorf("Expected builder.subsampleThreshold=%v: %v", expectedSubSampleThreshold, b.subsampleThreshold)
	}
}

func TestWord2vecTheta(t *testing.T) {
	b := &Word2vecBuilder{}

	expectedTheta := 1.0e-5
	b.Theta(expectedTheta)

	if b.theta != expectedTheta {
		t.Errorf("Expected builder.theta=%v: %v", expectedTheta, b.theta)
	}
}

func TestWord2vecInvalidModelBuild(t *testing.T) {
	b := &Word2vecBuilder{}

	b.Model(word2vec.ModelType(10))

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid model except for skip-gram|cbow: %v", b.model)
	}
}

func TestWord2vecInvalidOptimizerBuild(t *testing.T) {
	b := &Word2vecBuilder{}

	b.Optimizer(word2vec.OptimizerType(10))

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid optimizer except for ns|hs: %v", b.optimizer)
	}
}
