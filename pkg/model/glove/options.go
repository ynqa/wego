// Copyright © 2020 wego authors
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
package glove

import (
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"github.com/ynqa/wego/pkg/corpus"
	"github.com/ynqa/wego/pkg/corpus/pairwise"
	"github.com/ynqa/wego/pkg/model"
)

func invalidSolverTypeError(typ SolverType) error {
	return errors.Errorf("invalid solver: %s not in %s|%s", typ, Stochastic, AdaGrad)
}

type SolverType string

const (
	Stochastic        SolverType = "sgd"
	AdaGrad           SolverType = "adagrad"
	defaultSolverType            = Stochastic
)

func (t *SolverType) String() string {
	if *t == SolverType("") {
		*t = defaultSolverType
	}
	return string(*t)
}

func (t *SolverType) Set(name string) error {
	typ := SolverType(name)
	if typ == Stochastic || typ == AdaGrad {
		*t = typ
		return nil
	}
	return invalidSolverTypeError(typ)
}

func (t *SolverType) Type() string {
	return t.String()
}

const (
	defaultAlpha              = 0.75
	defaultSubsampleThreshold = 1.0e-3
	defaultXmax               = 100
)

type Options struct {
	CorpusOptions   corpus.Options
	PairwiseOptions pairwise.Options
	ModelOptions    model.Options

	Alpha              float64
	SolverType         SolverType
	SubsampleThreshold float64
	Xmax               int
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().Float64Var(&opts.Alpha, "alpha", defaultAlpha, "exponent of weighting function")
	cmd.Flags().Var(&opts.SolverType, "solver", "solver for GloVe objective. One of: sgd|adagrad")
	cmd.Flags().Float64Var(&opts.SubsampleThreshold, "threshold", defaultSubsampleThreshold, "threshold for subsampling")
	cmd.Flags().IntVar(&opts.Xmax, "xmax", defaultXmax, "specifying cutoff in weighting function")
}

type ModelOption func(*Options)

// corpus options
func WithMinCount(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.CorpusOptions.MinCount = v
	})
}

func ToLower() ModelOption {
	return ModelOption(func(opts *Options) {
		opts.CorpusOptions.ToLower = true
	})
}

// pairwise options
func WithCountType(typ pairwise.CountType) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.PairwiseOptions.CountType = typ
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

// for glove options
func WithAlpha(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Alpha = v
	})
}

func WithSolver(typ SolverType) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.SolverType = typ
	})
}

func WithSubsampleThreshold(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.SubsampleThreshold = v
	})
}

func WithXmax(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Xmax = v
	})
}
