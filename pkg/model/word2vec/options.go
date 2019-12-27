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

package word2vec

import (
	"fmt"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"github.com/ynqa/wego/pkg/corpus"
	"github.com/ynqa/wego/pkg/model"
)

func invalidModelTypeError(typ ModelType) error {
	return errors.Errorf("invalid model: %s not in %s|%s", typ, Cbow, SkipGram)
}
func invalidOptimizerTypeError(typ OptimizerType) error {
	return errors.Errorf("invalid optimizer: %s not in %s|%s", typ, NegativeSampling, HierarchicalSoftmax)
}

type ModelType string

const (
	Cbow             ModelType = "cbow"
	SkipGram         ModelType = "skipgram"
	defaultModelType           = Cbow
)

func (t *ModelType) String() string {
	if *t == ModelType("") {
		*t = defaultModelType
	}
	return string(*t)
}

func (t *ModelType) Set(name string) error {
	typ := ModelType(name)
	if typ == SkipGram || typ == Cbow {
		*t = typ
		return nil
	}
	return invalidModelTypeError(typ)
}

func (t *ModelType) Type() string {
	return t.String()
}

type OptimizerType string

const (
	NegativeSampling     OptimizerType = "ns"
	HierarchicalSoftmax  OptimizerType = "hs"
	defaultOptimizerType               = NegativeSampling
)

func (t *OptimizerType) String() string {
	if *t == OptimizerType("") {
		*t = defaultOptimizerType
	}
	return string(*t)
}

func (t *OptimizerType) Set(name string) error {
	typ := OptimizerType(name)
	if typ == NegativeSampling || typ == HierarchicalSoftmax {
		*t = typ
		return nil
	}
	return invalidOptimizerTypeError(typ)
}

func (t *OptimizerType) Type() string {
	return t.String()
}

const (
	defaultMaxDepth           = 100
	defaultNegativeSampleSize = 5
	defaultSubsampleThreshold = 1.0e-3
	defaultTheta              = 1.0e-4
)

type Options struct {
	CorpusOptions corpus.Options
	ModelOptions  model.Options

	MaxDepth           int
	ModelType          ModelType
	NegativeSampleSize int
	OptimizerType      OptimizerType
	SubsampleThreshold float64
	Theta              float64
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().IntVar(&opts.MaxDepth, "maxDepth", defaultMaxDepth, "times to track huffman tree, max-depth=0 means to track full path from root to word (for hierarchical softmax only)")
	cmd.Flags().Var(&opts.ModelType, "model", fmt.Sprintf("which model does it use? one of: %s|%s", Cbow, SkipGram))
	cmd.Flags().IntVar(&opts.NegativeSampleSize, "sample", defaultNegativeSampleSize, "negative sample size(for negative sampling only)")
	cmd.Flags().Var(&opts.OptimizerType, "optimizer", fmt.Sprintf("which optimizer does it use? one of: %s|%s", HierarchicalSoftmax, NegativeSampling))
	cmd.Flags().Float64Var(&opts.SubsampleThreshold, "threshold", defaultSubsampleThreshold, "threshold for subsampling")
	cmd.Flags().Float64Var(&opts.Theta, "theta", defaultTheta, "lower limit of learning rate (lr >= initlr * theta)")
}

type ModelOption func(*Options)

// corpus options
func ToLower() ModelOption {
	return ModelOption(func(opts *Options) {
		opts.CorpusOptions.ToLower = true
	})
}

// model options
func WithBatchSize(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.BatchSize = v
	})
}

func WithDimension(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Dim = v
	})
}

func WithInitLearningRate(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Initlr = v
	})
}

func WithIteration(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Iter = v
	})
}

func WithMinCount(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.MinCount = v
	})
}

func WithThreadSize(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.ThreadSize = v
	})
}

func WithWindow(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Window = v
	})
}

func Verbose() ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelOptions.Verbose = true
	})
}

// word2vec options
func WithMaxDepth(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.MaxDepth = v
	})
}

func WithModel(typ ModelType) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ModelType = typ
	})
}

func WithNegativeSampleSize(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.NegativeSampleSize = v
	})
}

func WithOptimizer(typ OptimizerType) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.OptimizerType = typ
	})
}

func WithSubsampleThreshold(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.SubsampleThreshold = v
	})
}

func WithTheta(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Theta = v
	})
}
