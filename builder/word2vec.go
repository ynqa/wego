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
	"github.com/pkg/errors"
	"github.com/spf13/viper"

	"github.com/ynqa/word-embedding/config"
	"github.com/ynqa/word-embedding/model"
	"github.com/ynqa/word-embedding/model/word2vec"
)

// Word2VecBuilder manages the members to build the Model interface.
// TODO: Validate the fields on Build called.
type Word2VecBuilder struct {
	dimension        int
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
		dimension:        config.DefaultDimension,
		window:           config.DefaultWindow,
		initLearningRate: config.DefaultInitLearningRate,
		dtype:            config.DefaultDtype,
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
		dimension:        viper.GetInt(config.Dimension.String()),
		window:           viper.GetInt(config.Window.String()),
		initLearningRate: viper.GetFloat64(config.InitLearningRate.String()),
		dtype:            viper.GetString(config.Dtype.String()),
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

// SetDimension sets the dimension of word vector.
func (wb *Word2VecBuilder) SetDimension(dimension int) *Word2VecBuilder {
	wb.dimension = dimension
	return wb
}

// SetWindow sets the context window size.
func (wb *Word2VecBuilder) SetWindow(window int) *Word2VecBuilder {
	wb.window = window
	return wb
}

// SetInitLearningRate sets the initial learning rate.
func (wb *Word2VecBuilder) SetInitLearningRate(initlr float64) *Word2VecBuilder {
	wb.initLearningRate = initlr
	return wb
}

// SetDtype sets the dtype for gorgonia tensor. One of: float32|float64
func (wb *Word2VecBuilder) SetDtype(dtype string) *Word2VecBuilder {
	wb.dtype = dtype
	return wb
}

// SetToLower converts the words in corpus to lowercase.
func (wb *Word2VecBuilder) SetToLower() *Word2VecBuilder {
	wb.toLower = true
	return wb
}

// SetVerbose sets verbose mode.
func (wb *Word2VecBuilder) SetVerbose() *Word2VecBuilder {
	wb.verbose = true
	return wb
}

// SetModel sets the model of Word2Vec. One of: cbow|skip-gram
func (wb *Word2VecBuilder) SetModel(model string) *Word2VecBuilder {
	wb.model = model
	return wb
}

// SetOptimizer sets the optimizer of Word2Vec. One of: hs|ns
func (wb *Word2VecBuilder) SetOptimizer(optimizer string) *Word2VecBuilder {
	wb.optimizer = optimizer
	return wb
}

// SetMaxDepth sets the number of times to track huffman tree.
func (wb *Word2VecBuilder) SetMaxDepth(maxDepth int) *Word2VecBuilder {
	wb.maxDepth = maxDepth
	return wb
}

// SetNegativeSampleSize sets the number of the samples as negative.
func (wb *Word2VecBuilder) SetNegativeSampleSize(size int) *Word2VecBuilder {
	wb.negativeSampleSize = size
	return wb
}

// SetTheta sets the lower limit of learning rate (lr >= initlr * theta).
func (wb *Word2VecBuilder) SetTheta(theta float64) *Word2VecBuilder {
	wb.theta = theta
	return wb
}

// SetBatchSize sets the batch size to update learning rate.
func (wb *Word2VecBuilder) SetBatchSize(batchSize int) *Word2VecBuilder {
	wb.batchSize = batchSize
	return wb
}

// SetSubSampleThreshold sets the threshold for subsampling.
func (wb *Word2VecBuilder) SetSubSampleThreshold(threshold float64) *Word2VecBuilder {
	wb.subsampleThreshold = threshold
	return wb
}

// Build creates model.Model interface.
func (wb *Word2VecBuilder) Build() (model.Model, error) {
	t, err := model.NewType(wb.dtype)
	if err != nil {
		return nil, err
	}

	cnf := model.NewConfig(wb.dimension, wb.window, wb.initLearningRate,
		t, wb.toLower, wb.verbose)

	var opt word2vec.Optimizer
	switch wb.optimizer {
	case "hs":
		opt = word2vec.NewHierarchicalSoftmax(wb.maxDepth)
	case "ns":
		opt = word2vec.NewNegativeSampling(wb.negativeSampleSize)
	default:
		return nil, errors.Errorf("Invalid optimizer: %s not in hs|ns", wb.optimizer)
	}

	state := word2vec.NewState(cnf, opt,
		wb.subsampleThreshold, wb.theta, wb.batchSize)

	switch wb.model {
	case "cbow":
		return word2vec.NewCBOW(state), nil
	case "skip-gram":
		return word2vec.NewSkipGram(state), nil
	default:
		return nil, errors.Errorf("Invalid model: %s not in cbow|skip-gram", wb.model)
	}
}
