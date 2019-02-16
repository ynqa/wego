// Copyright © 2019 Makoto Ito
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

	"github.com/ynqa/wego/config"
	"github.com/ynqa/wego/corpus"
	"github.com/ynqa/wego/model"
	"github.com/ynqa/wego/model/lexvec"
)

// LexvecBuilder manages the members to build Model interface.
type LexvecBuilder struct {
	// common configs.
	dimension      int
	iteration      int
	minCount       int
	threadSize     int
	batchSize      int
	window         int
	initlr         float64
	toLower        bool
	verbose        bool
	saveVectorType model.SaveVectorType

	// lexvec configs.
	negativeSampleSize int
	subsampleThreshold float64
	theta              float64
	smooth             float64
	relationType       corpus.RelationType
}

// NewLexvecBuilder creates *LexvecBuilder.
func NewLexvecBuilder() *LexvecBuilder {
	return &LexvecBuilder{
		dimension:      config.DefaultDimension,
		iteration:      config.DefaultIteration,
		minCount:       config.DefaultMinCount,
		threadSize:     config.DefaultThreadSize,
		batchSize:      config.DefaultBatchSize,
		window:         config.DefaultWindow,
		initlr:         config.DefaultInitlr,
		toLower:        config.DefaultToLower,
		verbose:        config.DefaultVerbose,
		saveVectorType: config.DefaultSaveVectorType,

		negativeSampleSize: config.DefaultNegativeSampleSize,
		subsampleThreshold: config.DefaultSubsampleThreshold,
		theta:              config.DefaultTheta,
		smooth:             config.DefaultSmooth,
		relationType:       config.DefaultRelationType,
	}
}

// NewLexvecBuilderFromViper creates *LexvecBuilder from viper.
func NewLexvecBuilderFromViper() (*LexvecBuilder, error) {
	var saveVectorType model.SaveVectorType
	saveVectorTypeStr := viper.GetString(config.SaveVectorType.String())
	switch saveVectorTypeStr {
	case model.NORMAL.String():
		saveVectorType = model.NORMAL
	case model.ADD.String():
		saveVectorType = model.ADD
	default:
		return nil, errors.Errorf("Invalid save vector type=%s", saveVectorTypeStr)
	}

	var relationType corpus.RelationType
	relationTypeStr := viper.GetString(config.RelationType.String())
	switch relationTypeStr {
	case corpus.PPMI.String():
		relationType = corpus.PPMI
	case corpus.PMI.String():
		relationType = corpus.PMI
	case corpus.CO.String():
		relationType = corpus.CO
	case corpus.LOGCO.String():
		relationType = corpus.LOGCO
	}

	return &LexvecBuilder{
		dimension:      viper.GetInt(config.Dimension.String()),
		iteration:      viper.GetInt(config.Iteration.String()),
		minCount:       viper.GetInt(config.MinCount.String()),
		threadSize:     viper.GetInt(config.ThreadSize.String()),
		batchSize:      viper.GetInt(config.BatchSize.String()),
		window:         viper.GetInt(config.Window.String()),
		initlr:         viper.GetFloat64(config.Initlr.String()),
		toLower:        viper.GetBool(config.ToLower.String()),
		verbose:        viper.GetBool(config.Verbose.String()),
		saveVectorType: saveVectorType,

		subsampleThreshold: viper.GetFloat64(config.SubsampleThreshold.String()),
		negativeSampleSize: viper.GetInt(config.NegativeSampleSize.String()),
		smooth:             viper.GetFloat64(config.Smooth.String()),
		relationType:       relationType,
	}, nil
}

// Dimension sets dimension of word vector.
func (lb *LexvecBuilder) Dimension(dimension int) *LexvecBuilder {
	lb.dimension = dimension
	return lb
}

// Iteration sets number of iteration.
func (lb *LexvecBuilder) Iteration(iter int) *LexvecBuilder {
	lb.iteration = iter
	return lb
}

// MinCount sets min count.
func (lb *LexvecBuilder) MinCount(minCount int) *LexvecBuilder {
	lb.minCount = minCount
	return lb
}

// ThreadSize sets number of goroutine.
func (lb *LexvecBuilder) ThreadSize(threadSize int) *LexvecBuilder {
	lb.threadSize = threadSize
	return lb
}

// BatchSize sets batch size to preprocess/train.
func (lb *LexvecBuilder) BatchSize(batchSize int) *LexvecBuilder {
	lb.batchSize = batchSize
	return lb
}

// Window sets context window size.
func (lb *LexvecBuilder) Window(window int) *LexvecBuilder {
	lb.window = window
	return lb
}

// Initlr sets initial learning rate.
func (lb *LexvecBuilder) Initlr(initlr float64) *LexvecBuilder {
	lb.initlr = initlr
	return lb
}

// ToLower is whether converts the words in corpus to lowercase or not.
func (lb *LexvecBuilder) ToLower() *LexvecBuilder {
	lb.toLower = true
	return lb
}

// Verbose sets verbose mode.
func (lb *LexvecBuilder) Verbose() *LexvecBuilder {
	lb.verbose = true
	return lb
}

func (lb *LexvecBuilder) SaveVectorType(typ model.SaveVectorType) *LexvecBuilder {
	lb.saveVectorType = typ
	return lb
}

// NegativeSampleSize sets number of samples as negative.
func (lb *LexvecBuilder) NegativeSampleSize(size int) *LexvecBuilder {
	lb.negativeSampleSize = size
	return lb
}

// SubSampleThreshold sets threshold for subsampling.
func (lb *LexvecBuilder) SubSampleThreshold(threshold float64) *LexvecBuilder {
	lb.subsampleThreshold = threshold
	return lb
}

func (lb *LexvecBuilder) Theta(theta float64) *LexvecBuilder {
	lb.theta = theta
	return lb
}

func (lb *LexvecBuilder) Smooth(smooth float64) *LexvecBuilder {
	lb.smooth = smooth
	return lb
}

func (lb *LexvecBuilder) RelationType(typ corpus.RelationType) *LexvecBuilder {
	lb.relationType = typ
	return lb
}

// Build creates Lexvec model.
func (lb *LexvecBuilder) Build() (model.Model, error) {
	o := &model.Option{
		Dimension:      lb.dimension,
		Iteration:      lb.iteration,
		MinCount:       lb.minCount,
		ThreadSize:     lb.threadSize,
		BatchSize:      lb.batchSize,
		Window:         lb.window,
		Initlr:         lb.initlr,
		ToLower:        lb.toLower,
		Verbose:        lb.verbose,
		SaveVectorType: lb.saveVectorType,
	}

	l := &lexvec.LexvecOption{
		NegativeSampleSize: lb.negativeSampleSize,
		SubSampleThreshold: lb.subsampleThreshold,
		Theta:              lb.theta,
		Smooth:             lb.smooth,
		RelationType:       lb.relationType,
	}

	return lexvec.NewLexvec(o, l), nil
}
