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

func TestGloveSetDimension(t *testing.T) {
	b := &GloveBuilder{}
	b.SetDimension(100)

	if b.dimension != 100 {
		t.Errorf("Expected builder.alpha=0.1: %v", b.alpha)
	}
}

func TestGloveSetWindow(t *testing.T) {
	b := &GloveBuilder{}
	b.SetWindow(10)

	if b.window != 10 {
		t.Errorf("Expected builder.window=10: %v", b.window)
	}
}

func TestGloveSetInitLearningRate(t *testing.T) {
	b := &GloveBuilder{}
	b.SetInitLearningRate(0.001)

	if b.initLearningRate != 0.001 {
		t.Errorf("Expected builder.initLearningRate=0.001: %v", b.initLearningRate)
	}
}

func TestGloveSetToLower(t *testing.T) {
	b := &GloveBuilder{}
	b.SetToLower()

	if !b.toLower {
		t.Errorf("Expected builder.lower=true: %v", b.toLower)
	}
}

func TestGloveSetVerbose(t *testing.T) {
	b := &GloveBuilder{}
	b.SetVerbose()

	if !b.verbose {
		t.Errorf("Expected builder.verbose=true: %v", b.verbose)
	}
}

func TestGloveSetSolver(t *testing.T) {
	b := &GloveBuilder{}
	b.SetSolver("adagrad")

	if b.solver != "adagrad" {
		t.Errorf("Expected builder.solver=adagrad: %v", b.solver)
	}
}

func TestGloveSetIteration(t *testing.T) {
	b := &GloveBuilder{}
	b.SetIteration(50)

	if b.iteration != 50 {
		t.Errorf("Expected builder.iteration=50: %v", b.iteration)
	}
}

func TestGloveSetAlpha(t *testing.T) {
	b := &GloveBuilder{}
	b.SetAlpha(0.1)

	if b.alpha != 0.1 {
		t.Errorf("Expected builder.alpha=0.1: %v", b.alpha)
	}
}

func TestGloveSetXmax(t *testing.T) {
	b := &GloveBuilder{}
	b.SetXmax(10)

	if b.xmax != 10 {
		t.Errorf("Expected builder.alpha=10: %v", b.xmax)
	}
}

func TestGloveSetMinCount(t *testing.T) {
	b := &GloveBuilder{}
	b.SetMinCount(10)

	if b.minCount != 10 {
		t.Errorf("Expected builder.minCount=10: %v", b.minCount)
	}
}

func TestGloveSetBatchSize(t *testing.T) {
	b := &GloveBuilder{}
	b.SetBatchSize(2048)

	if b.batchSize != 2048 {
		t.Errorf("Expected builder.batchSize=2048: %v", b.batchSize)
	}
}

func TestGloveInvalidSolverBuild(t *testing.T) {
	b := &GloveBuilder{}
	b.SetSolver("fake_solver")

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid solver except for sgd|adagrad: %v", b.solver)
	}
}
