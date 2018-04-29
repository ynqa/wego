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
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/viper"

	"github.com/ynqa/word-embedding/config"
	"github.com/ynqa/word-embedding/model"
	"github.com/ynqa/word-embedding/model/glove"
	"github.com/ynqa/word-embedding/validate"
)

// GloveBuilder manages the members to build Model interface.
type GloveBuilder struct {
	// input file path.
	inputFile string

	// common configs.
	dimension  int
	iteration  int
	minCount   int
	threadSize int
	window     int
	initlr     float64
	toLower    bool
	verbose    bool

	// glove configs.
	solver string
	xmax   int
	alpha  float64
}

// NewGloveBuilder creates *GloveBuilder
func NewGloveBuilder() *GloveBuilder {
	return &GloveBuilder{
		inputFile: config.DefaultInputFile,

		dimension:  config.DefaultDimension,
		iteration:  config.DefaultIteration,
		minCount:   config.DefaultMinCount,
		threadSize: config.DefaultThreadSize,
		window:     config.DefaultWindow,
		initlr:     config.DefaultInitlr,
		toLower:    config.DefaultToLower,
		verbose:    config.DefaultVerbose,

		solver: config.DefaultSolver,
		xmax:   config.DefaultXmax,
		alpha:  config.DefaultAlpha,
	}
}

// NewGloveBuilderFromViper creates *GloveBuilder from viper.
func NewGloveBuilderFromViper() *GloveBuilder {
	return &GloveBuilder{
		inputFile: viper.GetString(config.InputFile.String()),

		dimension:  viper.GetInt(config.Dimension.String()),
		iteration:  viper.GetInt(config.Iteration.String()),
		minCount:   viper.GetInt(config.MinCount.String()),
		threadSize: viper.GetInt(config.ThreadSize.String()),
		window:     viper.GetInt(config.Window.String()),
		initlr:     viper.GetFloat64(config.Initlr.String()),
		toLower:    viper.GetBool(config.ToLower.String()),
		verbose:    viper.GetBool(config.Verbose.String()),

		solver: viper.GetString(config.Solver.String()),
		xmax:   viper.GetInt(config.Xmax.String()),
		alpha:  viper.GetFloat64(config.Alpha.String()),
	}
}

// InputFile sets input file string.
func (gb *GloveBuilder) InputFile(inputFile string) *GloveBuilder {
	gb.inputFile = inputFile
	return gb
}

// Dimension sets dimension of word vector.
func (gb *GloveBuilder) Dimension(dimension int) *GloveBuilder {
	gb.dimension = dimension
	return gb
}

// Iteration sets number of iteration.
func (gb *GloveBuilder) Iteration(iter int) *GloveBuilder {
	gb.iteration = iter
	return gb
}

// MinCount sets min count.
func (gb *GloveBuilder) MinCount(minCount int) *GloveBuilder {
	gb.minCount = minCount
	return gb
}

// ThreadSize sets number of goroutine.
func (gb *GloveBuilder) ThreadSize(threadSize int) *GloveBuilder {
	gb.threadSize = threadSize
	return gb
}

// Window sets context window size.
func (gb *GloveBuilder) Window(window int) *GloveBuilder {
	gb.window = window
	return gb
}

// Initlr sets initial learning rate.
func (gb *GloveBuilder) Initlr(initlr float64) *GloveBuilder {
	gb.initlr = initlr
	return gb
}

// ToLower is whether converts the words in corpus to lowercase or not.
func (gb *GloveBuilder) ToLower() *GloveBuilder {
	gb.toLower = true
	return gb
}

// Verbose sets verbose mode.
func (gb *GloveBuilder) Verbose() *GloveBuilder {
	gb.verbose = true
	return gb
}

// Solver sets solver.
func (gb *GloveBuilder) Solver(solver string) *GloveBuilder {
	gb.solver = solver
	return gb
}

// Xmax sets x-max.
func (gb *GloveBuilder) Xmax(xmax int) *GloveBuilder {
	gb.xmax = xmax
	return gb
}

// Alpha sets alpha.
func (gb *GloveBuilder) Alpha(alpha float64) *GloveBuilder {
	gb.alpha = alpha
	return gb
}

// Build creates model.Model interface.
func (gb *GloveBuilder) Build() (model.Model, error) {
	if !validate.FileExists(gb.inputFile) {
		return nil, errors.Errorf("Not such a file %s", gb.inputFile)
	}

	input, err := os.Open(gb.inputFile)
	if err != nil {
		return nil, err
	}

	cnf := model.NewConfig(gb.dimension, gb.iteration, gb.minCount, gb.threadSize, gb.window,
		gb.initlr, gb.toLower, gb.verbose)

	var solver glove.Solver
	switch gb.solver {
	case "sgd":
		solver = glove.NewSgd(gb.dimension, gb.initlr)
	case "adagrad":
		solver = glove.NewAdaGrad(gb.dimension, gb.initlr)
	default:
		return nil, errors.Errorf("Invalid solver: %s not in sgd|adagrad", gb.solver)
	}

	return glove.NewGlove(input, cnf, solver, gb.xmax, gb.alpha)
}
