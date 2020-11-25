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
	"fmt"
	"runtime"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	co "github.com/ynqa/wego/pkg/corpus/cooccurrence"
)

func invalidSolverTypeError(typ SolverType) error {
	return errors.Errorf("invalid solver: %s not in %s|%s", typ, Stochastic, AdaGrad)
}

type SolverType string

const (
	Stochastic SolverType = "sgd"
	AdaGrad    SolverType = "adagrad"
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

var (
	defaultAlpha              = 0.75
	defaultBatchSize          = 100000
	defaultDim                = 10
	defaultDocInMemory        = false
	defaultGoroutines         = runtime.NumCPU()
	defaultInitlr             = 0.025
	defaultIter               = 15
	defaultMaxCount           = -1
	defaultMinCount           = 5
	defaultSolverType         = Stochastic
	defaultSubsampleThreshold = 1.0e-3
	defaultToLower            = false
	defaultVerbose            = false
	defaultWindow             = 5
	defaultXmax               = 100
)

type Options struct {
	Alpha              float64
	BatchSize          int
	CountType          co.CountType
	Dim                int
	DocInMemory        bool
	Goroutines         int
	Initlr             float64
	Iter               int
	MaxCount           int
	MinCount           int
	SolverType         SolverType
	SubsampleThreshold float64
	ToLower            bool
	Verbose            bool
	Window             int
	Xmax               int
}

func DefaultOptions() Options {
	return Options{
		Alpha:              defaultAlpha,
		BatchSize:          defaultBatchSize,
		CountType:          co.DefaultCountType,
		Dim:                defaultDim,
		DocInMemory:        defaultDocInMemory,
		Goroutines:         defaultGoroutines,
		Initlr:             defaultInitlr,
		Iter:               defaultIter,
		MaxCount:           defaultMaxCount,
		MinCount:           defaultMinCount,
		SolverType:         defaultSolverType,
		SubsampleThreshold: defaultSubsampleThreshold,
		ToLower:            defaultToLower,
		Verbose:            defaultVerbose,
		Window:             defaultWindow,
		Xmax:               defaultXmax,
	}
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().Float64Var(&opts.Alpha, "alpha", defaultAlpha, "exponent of weighting function")
	cmd.Flags().IntVar(&opts.BatchSize, "batch", defaultBatchSize, "batch size to train")
	cmd.Flags().Var(&opts.CountType, "cnt", fmt.Sprintf("count type for co-occurrence words. One of %s|%s", co.Increment, co.Proximity))
	cmd.Flags().IntVarP(&opts.Dim, "dim", "d", defaultDim, "dimension for word vector")
	cmd.Flags().IntVar(&opts.Goroutines, "goroutines", defaultGoroutines, "number of goroutine")
	cmd.Flags().BoolVar(&opts.DocInMemory, "in-memory", defaultDocInMemory, "whether to store the doc in memory")
	cmd.Flags().Float64Var(&opts.Initlr, "initlr", defaultInitlr, "initial learning rate")
	cmd.Flags().IntVar(&opts.Iter, "iter", defaultIter, "number of iteration")
	cmd.Flags().IntVar(&opts.MaxCount, "max-count", defaultMaxCount, "upper limit to filter words")
	cmd.Flags().IntVar(&opts.MinCount, "min-count", defaultMinCount, "lower limit to filter words")

	cmd.Flags().Var(&opts.SolverType, "solver", "solver for GloVe objective. One of: sgd|adagrad")
	cmd.Flags().Float64Var(&opts.SubsampleThreshold, "threshold", defaultSubsampleThreshold, "threshold for subsampling")
	cmd.Flags().BoolVar(&opts.ToLower, "to-lower", defaultToLower, "whether the words on corpus convert to lowercase or not")
	cmd.Flags().BoolVar(&opts.Verbose, "verbose", defaultVerbose, "verbose mode")
	cmd.Flags().IntVarP(&opts.Window, "window", "w", defaultWindow, "context window size")

	cmd.Flags().IntVar(&opts.Xmax, "xmax", defaultXmax, "specifying cutoff in weighting function")
}

type ModelOption func(*Options)

func Alpha(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Alpha = v
	})
}

func BatchSize(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.BatchSize = v
	})
}

func DocInMemory() ModelOption {
	return ModelOption(func(opts *Options) {
		opts.DocInMemory = true
	})
}

func Goroutines(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Goroutines = v
	})
}

func Dim(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Dim = v
	})
}

func Initlr(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Initlr = v
	})
}

func Iter(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Iter = v
	})
}

func MaxCount(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.MaxCount = v
	})
}

func MinCount(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.MinCount = v
	})
}

func Solver(typ SolverType) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.SolverType = typ
	})
}

func SubsampleThreshold(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.SubsampleThreshold = v
	})
}

func ToLower() ModelOption {
	return ModelOption(func(opts *Options) {
		opts.ToLower = true
	})
}

func Verbose() ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Verbose = true
	})
}

func Window(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Window = v
	})
}

func Xmax(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Xmax = v
	})
}
