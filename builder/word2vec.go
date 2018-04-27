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
	"github.com/ynqa/word-embedding/model/word2vec"
	"github.com/ynqa/word-embedding/validate"
)

// Word2VecBuilder manages the members to build the Model interface.
// TODO: Validate the fields on Build called.
type Word2VecBuilder struct {
	inputFile string

	dimension        int
	iteration        int
	thread           int
	minCount         int
	window           int
	initLearningRate float64
	dtype            string
	toLower          bool
	verbose          bool

	model              string
	optimizer          string
	maxDepth           int
	negativeSampleSize int
	theta              float64
	batchSize          int
	subsampleThreshold float64
}

// NewWord2VecBuilder creates *Word2VecBuilder.
func NewWord2VecBuilder() *Word2VecBuilder {
	return &Word2VecBuilder{
		inputFile: config.DefaultInputFile,

		dimension:        config.DefaultDimension,
		iteration:        config.DefaultIteration,
		minCount:         config.DefaultMinCount,
		thread:           config.DefaultThread,
		window:           config.DefaultWindow,
		initLearningRate: config.DefaultInitLearningRate,
		toLower:          config.DefaultToLower,
		verbose:          config.DefaultVerbose,

		model:              config.DefaultModel,
		optimizer:          config.DefaultOptimizer,
		maxDepth:           config.DefaultMaxDepth,
		negativeSampleSize: config.DefaultNegativeSampleSize,
		theta:              config.DefaultTheta,
		batchSize:          config.DefaultBatchSize,
		subsampleThreshold: config.DefaultSubsampleThreshold,
	}
}

// NewWord2VecBuilderViper creates *Word2VecBuilder using viper.
func NewWord2VecBuilderViper() *Word2VecBuilder {
	return &Word2VecBuilder{
		inputFile: viper.GetString(config.InputFile.String()),

		dimension:        viper.GetInt(config.Dimension.String()),
		iteration:        viper.GetInt(config.Iteration.String()),
		minCount:         viper.GetInt(config.MinCount.String()),
		thread:           viper.GetInt(config.Thread.String()),
		window:           viper.GetInt(config.Window.String()),
		initLearningRate: viper.GetFloat64(config.InitLearningRate.String()),
		toLower:          viper.GetBool(config.ToLower.String()),
		verbose:          viper.GetBool(config.Verbose.String()),

		model:              viper.GetString(config.Model.String()),
		optimizer:          viper.GetString(config.Optimizer.String()),
		maxDepth:           viper.GetInt(config.MaxDepth.String()),
		negativeSampleSize: viper.GetInt(config.NegativeSampleSize.String()),
		theta:              viper.GetFloat64(config.Theta.String()),
		batchSize:          viper.GetInt(config.BatchSize.String()),
		subsampleThreshold: viper.GetFloat64(config.SubsampleThreshold.String()),
	}
}

// InputFile sets the input file string.
func (wb *Word2VecBuilder) InputFile(inputFile string) *Word2VecBuilder {
	wb.inputFile = inputFile
	return wb
}

// Dimension sets the dimension of word vector.
func (wb *Word2VecBuilder) Dimension(dimension int) *Word2VecBuilder {
	wb.dimension = dimension
	return wb
}

// Iteration sets the number of iteration.
func (wb *Word2VecBuilder) Iteration(iter int) *Word2VecBuilder {
	wb.iteration = iter
	return wb
}

// MinCount sets min count.
func (wb *Word2VecBuilder) MinCount(minCount int) *Word2VecBuilder {
	wb.minCount = minCount
	return wb
}

// Thread sets number of goroutine.
func (wb *Word2VecBuilder) Thread(thread int) *Word2VecBuilder {
	wb.thread = thread
	return wb
}

// Window sets the context window size.
func (wb *Word2VecBuilder) Window(window int) *Word2VecBuilder {
	wb.window = window
	return wb
}

// InitLearningRate sets the initial learning rate.
func (wb *Word2VecBuilder) InitLearningRate(initlr float64) *Word2VecBuilder {
	wb.initLearningRate = initlr
	return wb
}

// ToLower converts the words in corpus to lowercase.
func (wb *Word2VecBuilder) ToLower() *Word2VecBuilder {
	wb.toLower = true
	return wb
}

// Verbose sets verbose mode.
func (wb *Word2VecBuilder) Verbose() *Word2VecBuilder {
	wb.verbose = true
	return wb
}

// Model sets the model of Word2Vec. One of: cbow|skip-gram
func (wb *Word2VecBuilder) Model(model string) *Word2VecBuilder {
	wb.model = model
	return wb
}

// Optimizer sets the optimizer of Word2Vec. One of: hs|ns
func (wb *Word2VecBuilder) Optimizer(optimizer string) *Word2VecBuilder {
	wb.optimizer = optimizer
	return wb
}

// MaxDepth sets the number of times to track huffman tree.
func (wb *Word2VecBuilder) MaxDepth(maxDepth int) *Word2VecBuilder {
	wb.maxDepth = maxDepth
	return wb
}

// NegativeSampleSize sets the number of the samples as negative.
func (wb *Word2VecBuilder) NegativeSampleSize(size int) *Word2VecBuilder {
	wb.negativeSampleSize = size
	return wb
}

// Theta sets the lower limit of learning rate (lr >= initlr * theta).
func (wb *Word2VecBuilder) Theta(theta float64) *Word2VecBuilder {
	wb.theta = theta
	return wb
}

// BatchSize sets the batch size to update learning rate.
func (wb *Word2VecBuilder) BatchSize(batchSize int) *Word2VecBuilder {
	wb.batchSize = batchSize
	return wb
}

// SubSampleThreshold sets the threshold for subsampling.
func (wb *Word2VecBuilder) SubSampleThreshold(threshold float64) *Word2VecBuilder {
	wb.subsampleThreshold = threshold
	return wb
}

// Build creates model.Model interface.
func (wb *Word2VecBuilder) Build() (model.Model, error) {
	if !validate.FileExists(wb.inputFile) {
		return nil, errors.Errorf("Not such a file %s", wb.inputFile)
	}

	input, err := os.Open(wb.inputFile)
	if err != nil {
		return nil, err
	}

	cnf := model.NewConfig(wb.dimension, wb.iteration, wb.minCount, wb.thread, wb.window,
		wb.initLearningRate, wb.toLower, wb.verbose)

	var opt word2vec.Optimizer
	switch wb.optimizer {
	case "hs":
		opt = word2vec.NewHierarchicalSoftmax(wb.maxDepth)
	case "ns":
		opt = word2vec.NewNegativeSampling(wb.negativeSampleSize)
	default:
		return nil, errors.Errorf("Invalid optimizer: %s not in hs|ns", wb.optimizer)
	}

	var mod word2vec.Model
	switch wb.model {
	case "cbow":
		mod = word2vec.NewCBOW(cnf)
	case "skip-gram":
		mod = word2vec.NewSkipGram(cnf)
	default:
		return nil, errors.Errorf("Invalid model: %s not in cbow|skip-gram", wb.model)
	}

	return word2vec.NewWord2Vec(input, cnf, mod, opt,
		wb.subsampleThreshold, wb.theta, wb.batchSize), nil
}
