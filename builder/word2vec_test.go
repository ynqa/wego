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

func TestSetDimension(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetDimension(100)

	if b.dimension != 100 {
		t.Errorf("Expected builder.dimension=100: %v", b.dimension)
	}
}

func TestSetWindow(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetWindow(10)

	if b.window != 10 {
		t.Errorf("Expected builder.window=10: %v", b.window)
	}
}

func TestSetInitLearningRate(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetInitLearningRate(0.001)

	if b.initLearningRate != 0.001 {
		t.Errorf("Expected builder.initLearningRate=0.001: %v", b.initLearningRate)
	}
}

func TestSetDtype(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetDtype("float32")

	if b.dtype != "float32" {
		t.Errorf("Expected builder.dtype=float32: %v", b.dtype)
	}
}

func TestSetToLower(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetToLower()

	if !b.toLower {
		t.Errorf("Expected builder.lower=true: %v", b.toLower)
	}
}

func TestSetVerbose(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetVerbose()

	if !b.verbose {
		t.Errorf("Expected builder.verbose=true: %v", b.verbose)
	}
}

func TestSetModel(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetModel("skip-gram")

	if b.model != "skip-gram" {
		t.Errorf("Expected builder.model=skip-gram: %v", b.model)
	}
}

func TestSetOptimizer(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetOptimizer("ns")

	if b.optimizer != "ns" {
		t.Errorf("Expected builder.optimizer=ns: %v", b.optimizer)
	}
}

func TestSetMaxDepth(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetMaxDepth(40)

	if b.maxDepth != 40 {
		t.Errorf("Expected builder.maxDepth=40: %v", b.maxDepth)
	}
}

func TestSetNegativeSampleSize(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetNegativeSampleSize(20)

	if b.negativeSampleSize != 20 {
		t.Errorf("Expected builder.negativeSampleSize=20: %v", b.negativeSampleSize)
	}
}

func TestSetTheta(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetTheta(1.0e-5)

	if b.theta != 1.0e-5 {
		t.Errorf("Expected builder.theta=1.0e-5: %v", b.theta)
	}
}

func TestSetBatchSize(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetBatchSize(2048)

	if b.batchSize != 2048 {
		t.Errorf("Expected builder.batchSize=2048: %v", b.batchSize)
	}
}

func TestSetSubSampleThreshold(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetSubSampleThreshold(0.001)

	if b.subsampleThreshold != 0.001 {
		t.Errorf("Expected builder.subsampleThreshold=0.001: %v", b.subsampleThreshold)
	}
}

func TestInvalidTypeBuild(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetDtype("fake_dtype")

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid dtype except for skip-gram|cbow: %v", b.dtype)
	}
}

func TestInvalidModelBuild(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetModel("fake_model")

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid model except for skip-gram|cbow: %v", b.model)
	}
}

func TestInvalidOptimizerBuild(t *testing.T) {
	b := &Word2VecBuilder{}
	b.SetOptimizer("fake_optimizer")

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid optimizer except for ns|hs: %v", b.optimizer)
	}
}
