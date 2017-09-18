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
	builder := &Word2VecBuilder{}
	builder.SetDimension(100)

	if builder.dimension != 100 {
		t.Errorf("Expected builder.dimension=100: %v", builder.dimension)
	}
}

func TestSetWindow(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetWindow(10)

	if builder.window != 10 {
		t.Errorf("Expected builder.window=10: %v", builder.window)
	}
}

func TestSetInitLearningRate(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetInitLearningRate(0.001)

	if builder.initLearningRate != 0.001 {
		t.Errorf("Expected builder.initLearningRate=0.001: %v", builder.initLearningRate)
	}
}

func TestSetLower(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetLower(false)

	if builder.lower {
		t.Errorf("Expected builder.lower=false: %v", builder.lower)
	}
}

func TestSetModel(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetModel("skip-gram")

	if builder.model != "skip-gram" {
		t.Errorf("Expected builder.model=skip-gram: %v", builder.model)
	}
}

func TestSetOptimizer(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetOptimizer("ns")

	if builder.optimizer != "ns" {
		t.Errorf("Expected builder.optimizer=ns: %v", builder.optimizer)
	}
}

func TestSetMaxDepth(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetMaxDepth(40)

	if builder.maxDepth != 40 {
		t.Errorf("Expected builder.maxDepth=40: %v", builder.maxDepth)
	}
}

func TestSetNegativeSampleSize(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetNegativeSampleSize(20)

	if builder.negativeSampleSize != 20 {
		t.Errorf("Expected builder.negativeSampleSize=20: %v", builder.negativeSampleSize)
	}
}

func TestSetTheta(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetTheta(1.0e-5)

	if builder.theta != 1.0e-5 {
		t.Errorf("Expected builder.theta=1.0e-5: %v", builder.theta)
	}
}

func TestSetBatchSize(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetBatchSize(2048)

	if builder.batchSize != 2048 {
		t.Errorf("Expected builder.batchSize=2048: %v", builder.batchSize)
	}
}

func TestSetSubSampleThreshold(t *testing.T) {
	builder := &Word2VecBuilder{}
	builder.SetSubSampleThreshold(0.001)

	if builder.subsampleThreshold != 0.001 {
		t.Errorf("Expected builder.subsampleThreshold=0.001: %v", builder.subsampleThreshold)
	}
}

func TestInvalidModelBuild(t *testing.T) {
	builder := NewWord2VecBuilder()
	builder.SetModel("fake_model")

	if _, err := builder.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid model in skip-gram|cbow: %v", builder.model)
	}
}

func TestInvalidOptimizerBuild(t *testing.T) {
	builder := NewWord2VecBuilder()
	builder.SetOptimizer("fake_optimizer")

	if _, err := builder.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid optimizer in ns|hs: %v", builder.optimizer)
	}
}

func TestString(t *testing.T) {
	builder := NewWord2VecBuilder()

	if builder.String() == "" {
		t.Error("String() in Word2VecBuilder shows null")
	}
}
