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
	"github.com/spf13/viper"

	"github.com/ynqa/word-embedding/config"
	"github.com/ynqa/word-embedding/model"
	"github.com/ynqa/word-embedding/model/glove"
)

// GloVeBuilder manages the members to build the Model interface.
type GloVeBuilder struct {
	dimension        int
	window           int
	initLearningRate float64
	thread           int
	dtype            string
	toLower          bool
	verbose          bool

	iteration int
	alpha     float64
	xmax      int
}

// NewGloVeBuilder creates *GloVeBuilder
func NewGloVeBuilder() *GloVeBuilder {
	return &GloVeBuilder{
		dimension:        config.DefaultDimension,
		window:           config.DefaultWindow,
		initLearningRate: config.DefaultInitLearningRate,
		thread:           config.DefaultThread,
		toLower:          config.DefaultToLower,
		verbose:          config.DefaultVerbose,

		iteration: config.DefaultIteration,
		alpha:     config.DefaultAlpha,
		xmax:      config.DefaultXmax,
	}
}

// NewGloVeBuilderViper creates *GloVeBuilder using viper.
func NewGloVeBuilderViper() *GloVeBuilder {
	return &GloVeBuilder{
		dimension:        viper.GetInt(config.Dimension.String()),
		window:           viper.GetInt(config.Window.String()),
		initLearningRate: viper.GetFloat64(config.InitLearningRate.String()),
		thread:           viper.GetInt(config.Thread.String()),
		toLower:          viper.GetBool(config.ToLower.String()),
		verbose:          viper.GetBool(config.Verbose.String()),

		iteration: viper.GetInt(config.Iteration.String()),
		alpha:     viper.GetFloat64(config.Alpha.String()),
		xmax:      viper.GetInt(config.Xmax.String()),
	}
}

// SetDimension sets the dimension of word vector.
func (gb *GloVeBuilder) SetDimension(dimension int) *GloVeBuilder {
	gb.dimension = dimension
	return gb
}

// SetWindow sets the context window size.
func (gb *GloVeBuilder) SetWindow(window int) *GloVeBuilder {
	gb.window = window
	return gb
}

// SetInitLearningRate sets the initial learning rate.
func (gb *GloVeBuilder) SetInitLearningRate(initlr float64) *GloVeBuilder {
	gb.initLearningRate = initlr
	return gb
}

// SetThread sets number of goroutine.
func (gb *GloVeBuilder) SetThread(thread int) *GloVeBuilder {
	gb.thread = thread
	return gb
}

// SetDtype sets the dtype for gorgonia tensor. One of: float32|float64
func (gb *GloVeBuilder) SetDtype(dtype string) *GloVeBuilder {
	gb.dtype = dtype
	return gb
}

// SetToLower converts the words in corpus to lowercase.
func (gb *GloVeBuilder) SetToLower() *GloVeBuilder {
	gb.toLower = true
	return gb
}

// SetVerbose sets verbose mode.
func (gb *GloVeBuilder) SetVerbose() *GloVeBuilder {
	gb.verbose = true
	return gb
}

// SetIteration sets the number of iteration.
func (gb *GloVeBuilder) SetIteration(iter int) *GloVeBuilder {
	gb.iteration = iter
	return gb
}

// SetAlpha sets alpha.
func (gb *GloVeBuilder) SetAlpha(alpha float64) *GloVeBuilder {
	gb.alpha = alpha
	return gb
}

// SetXmax sets Xmax.
func (gb *GloVeBuilder) SetXmax(xmax int) *GloVeBuilder {
	gb.xmax = xmax
	return gb
}

// Build creates model.Model interface.
func (gb *GloVeBuilder) Build() (model.Model, error) {
	cnf := model.NewConfig(gb.dimension, gb.window, gb.initLearningRate,
		gb.thread, gb.toLower, gb.verbose)

	return glove.NewGloVe(cnf, gb.iteration, gb.alpha, gb.xmax), nil
}
