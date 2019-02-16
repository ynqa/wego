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

	"github.com/ynqa/wego/model/glove"
)

func TestGloveDimension(t *testing.T) {
	b := &GloveBuilder{}

	expectedDimension := 100
	b.Dimension(expectedDimension)

	if b.dimension != expectedDimension {
		t.Errorf("Expected builder.dimension=%v: %v", expectedDimension, b.dimension)
	}
}

func TestGloveIteration(t *testing.T) {
	b := &GloveBuilder{}

	expectedIteration := 50
	b.Iteration(expectedIteration)

	if b.iteration != expectedIteration {
		t.Errorf("Expected builder.iteration=%v: %v", expectedIteration, b.iteration)
	}
}

func TestGloveMinCount(t *testing.T) {
	b := &GloveBuilder{}

	expectedMinCount := 10
	b.MinCount(expectedMinCount)

	if b.minCount != expectedMinCount {
		t.Errorf("Expected builder.minCount=%v: %v", expectedMinCount, b.minCount)
	}
}

func TestGloveThreadSize(t *testing.T) {
	b := &GloveBuilder{}

	expectedThreadSize := 8
	b.ThreadSize(expectedThreadSize)

	if b.threadSize != expectedThreadSize {
		t.Errorf("Expected builder.threadSize=%v: %v", expectedThreadSize, b.threadSize)
	}
}

func TestGloveWindow(t *testing.T) {
	b := &GloveBuilder{}

	expectedWindow := 10
	b.Window(expectedWindow)

	if b.window != expectedWindow {
		t.Errorf("Expected builder.window=%v: %v", expectedWindow, b.window)
	}
}

func TestGloveInitlr(t *testing.T) {
	b := &GloveBuilder{}

	expectedInitlr := 0.001
	b.Initlr(expectedInitlr)

	if b.initlr != expectedInitlr {
		t.Errorf("Expected builder.initlr=%v: %v", expectedInitlr, b.initlr)
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

	expectedSolver := glove.ADAGRAD
	b.Solver(expectedSolver)

	if b.solver != expectedSolver {
		t.Errorf("Expected builder.solver=%v: %v", expectedSolver, b.solver)
	}
}

func TestGloveXmax(t *testing.T) {
	b := &GloveBuilder{}

	expectedXmax := 10
	b.Xmax(expectedXmax)

	if b.xmax != expectedXmax {
		t.Errorf("Expected builder.xmax=%v: %v", expectedXmax, b.xmax)
	}
}

func TestGloveAlpha(t *testing.T) {
	b := &GloveBuilder{}

	exoectedAlpha := 0.1
	b.Alpha(exoectedAlpha)

	if b.alpha != exoectedAlpha {
		t.Errorf("Expected builder.alpha=%v: %v", exoectedAlpha, b.alpha)
	}
}

func TestGloveInvalidSolverBuild(t *testing.T) {
	b := &GloveBuilder{}

	b.Solver(glove.SolverType(10))

	if _, err := b.Build(); err == nil {
		t.Errorf("Expected to fail building with invalid solver except for sgd|adagrad: %v", b.solver)
	}
}
