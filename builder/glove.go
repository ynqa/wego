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

// GloveBuilder manages the members to build the Model interface.
type GloveBuilder struct {
	inputFile string

	dimension        int
	iteration        int
	minCount         int
	thread           int
	window           int
	initLearningRate float64
	toLower          bool
	verbose          bool

	solver string
	alpha  float64
	xmax   int
}

// NewGloveBuilder creates *GloveBuilder
func NewGloveBuilder() *GloveBuilder {
	return &GloveBuilder{
		inputFile: config.DefaultInputFile,

		dimension:        config.DefaultDimension,
		iteration:        config.DefaultIteration,
		minCount:         config.DefaultMinCount,
		thread:           config.DefaultThread,
		window:           config.DefaultWindow,
		initLearningRate: config.DefaultInitLearningRate,
		toLower:          config.DefaultToLower,
		verbose:          config.DefaultVerbose,

		solver: config.DefaultSolver,
		alpha:  config.DefaultAlpha,
		xmax:   config.DefaultXmax,
	}
}

// NewGloveBuilderViper creates *GloveBuilder using viper.
func NewGloveBuilderViper() *GloveBuilder {
	return &GloveBuilder{
		inputFile: viper.GetString(config.InputFile.String()),

		dimension:        viper.GetInt(config.Dimension.String()),
		iteration:        viper.GetInt(config.Iteration.String()),
		minCount:         viper.GetInt(config.MinCount.String()),
		thread:           viper.GetInt(config.Thread.String()),
		window:           viper.GetInt(config.Window.String()),
		initLearningRate: viper.GetFloat64(config.InitLearningRate.String()),
		toLower:          viper.GetBool(config.ToLower.String()),
		verbose:          viper.GetBool(config.Verbose.String()),

		solver: viper.GetString(config.Solver.String()),
		alpha:  viper.GetFloat64(config.Alpha.String()),
		xmax:   viper.GetInt(config.Xmax.String()),
	}
}

// InputFile sets the input file string.
func (gb *GloveBuilder) InputFile(inputFile string) *GloveBuilder {
	gb.inputFile = inputFile
	return gb
}

// Dimension sets the dimension of word vector.
func (gb *GloveBuilder) Dimension(dimension int) *GloveBuilder {
	gb.dimension = dimension
	return gb
}

// Iteration sets the number of iteration.
func (gb *GloveBuilder) Iteration(iter int) *GloveBuilder {
	gb.iteration = iter
	return gb
}

// MinCount sets min count.
func (gb *GloveBuilder) MinCount(minCount int) *GloveBuilder {
	gb.minCount = minCount
	return gb
}

// Thread sets number of goroutine.
func (gb *GloveBuilder) Thread(thread int) *GloveBuilder {
	gb.thread = thread
	return gb
}

// Window sets the context window size.
func (gb *GloveBuilder) Window(window int) *GloveBuilder {
	gb.window = window
	return gb
}

// InitLearningRate sets the initial learning rate.
func (gb *GloveBuilder) InitLearningRate(initlr float64) *GloveBuilder {
	gb.initLearningRate = initlr
	return gb
}

// ToLower converts the words in corpus to lowercase.
func (gb *GloveBuilder) ToLower() *GloveBuilder {
	gb.toLower = true
	return gb
}

// Verbose sets verbose mode.
func (gb *GloveBuilder) Verbose() *GloveBuilder {
	gb.verbose = true
	return gb
}

// Solver sets the solver.
func (gb *GloveBuilder) Solver(solver string) *GloveBuilder {
	gb.solver = solver
	return gb
}

// Alpha sets alpha.
func (gb *GloveBuilder) Alpha(alpha float64) *GloveBuilder {
	gb.alpha = alpha
	return gb
}

// Xmax sets x-max.
func (gb *GloveBuilder) Xmax(xmax int) *GloveBuilder {
	gb.xmax = xmax
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

	cnf := model.NewConfig(gb.dimension, gb.iteration, gb.minCount, gb.thread, gb.window,
		gb.initLearningRate, gb.toLower, gb.verbose)

	var solver glove.Solver
	switch gb.solver {
	case "sgd":
		solver = glove.NewSGD(cnf)
	case "adagrad":
		solver = glove.NewAdaGrad(cnf)
	default:
		return nil, errors.Errorf("Invalid solver: %s not in sgd|adagrad", gb.solver)
	}

	return glove.NewGlove(input, cnf, solver, gb.xmax, gb.alpha), nil
}
