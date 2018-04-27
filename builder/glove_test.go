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

func TestGloveInputFile(t *testing.T) {
	b := &GloveBuilder{}
	b.InputFile("inputfile")

	if b.inputFile != "inputfile" {
		t.Errorf("Expected builder.inputFile=inputfile: %v", b.inputFile)
	}
}

func TestGloveDimension(t *testing.T) {
	b := &GloveBuilder{}
	b.Dimension(100)

	if b.dimension != 100 {
		t.Errorf("Expected builder.alpha=0.1: %v", b.alpha)
	}
}

func TestGloveIteration(t *testing.T) {
	b := &GloveBuilder{}
	b.Iteration(50)

	if b.iteration != 50 {
		t.Errorf("Expected builder.iteration=50: %v", b.iteration)
	}
}

func TestGloveMinCount(t *testing.T) {
	b := &GloveBuilder{}
	b.MinCount(10)

	if b.minCount != 10 {
		t.Errorf("Expected builder.minCount=10: %v", b.minCount)
	}
}

func TestGloveWindow(t *testing.T) {
	b := &GloveBuilder{}
	b.Window(10)

	if b.window != 10 {
		t.Errorf("Expected builder.window=10: %v", b.window)
	}
}

func TestGloveInitLearningRate(t *testing.T) {
	b := &GloveBuilder{}
	b.InitLearningRate(0.001)

	if b.initLearningRate != 0.001 {
		t.Errorf("Expected builder.initLearningRate=0.001: %v", b.initLearningRate)
	}
}

func TestGloveToLower(t *testing.T) {
	b := &GloveBuilder{}
	b.ToLower()

	if !b.toLower {
		t.Errorf("Expected builder.lower=true: %v", b.toLower)
	}
}

func TestGloveVerbose(t *testing.T) {
	b := &GloveBuilder{}
	b.Verbose()

	if !b.verbose {
		t.Errorf("Expected builder.verbose=true: %v", b.verbose)
	}
}

func TestGloveSolver(t *testing.T) {
	b := &GloveBuilder{}
	b.Solver("adagrad")

	if b.solver != "adagrad" {
		t.Errorf("Expected builder.solver=adagrad: %v", b.solver)
	}
}

func TestGloveAlpha(t *testing.T) {
	b := &GloveBuilder{}
	b.Alpha(0.1)

	if b.alpha != 0.1 {
		t.Errorf("Expected builder.alpha=0.1: %v", b.alpha)
	}
}

func TestGloveXmax(t *testing.T) {
	b := &GloveBuilder{}
	b.Xmax(10)

	if b.xmax != 10 {
		t.Errorf("Expected builder.alpha=10: %v", b.xmax)
	}
}

func TestGloveInvalidSolverBuild(t *testing.T) {
	b := &GloveBuilder{}
	b.Solver("fake_solver")

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid solver except for sgd|adagrad: %v", b.solver)
	}
}
