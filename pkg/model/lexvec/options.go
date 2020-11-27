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

package lexvec

import (
	"fmt"
	"runtime"

	"github.com/spf13/cobra"
)

type RelationType = string

const (
	PPMI           RelationType = "ppmi"
	PMI            RelationType = "pmi"
	Collocation    RelationType = "co"
	LogCollocation RelationType = "logco"
)

var (
	defaultBatchSize          = 10000
	defaultDim                = 10
	defaultDocInMemory        = false
	defaultGoroutines         = runtime.NumCPU()
	defaultInitlr             = 0.025
	defaultIter               = 15
	defaultLogBatch           = 100000
	defaultMaxCount           = -1
	defaultMinCount           = 5
	defaultMinLR              = defaultInitlr * 1.0e-4
	defaultNegativeSampleSize = 5
	defaultRelationType       = PPMI
	defaultSmooth             = 0.75
	defaultSubsampleThreshold = 1.0e-3
	defaultToLower            = false
	defaultUpdateLRBatch      = 100000
	defaultVerbose            = false
	defaultWindow             = 5
)

type Options struct {
	BatchSize          int
	Dim                int
	DocInMemory        bool
	Goroutines         int
	Initlr             float64
	Iter               int
	LogBatch           int
	MaxCount           int
	MinCount           int
	MinLR              float64
	NegativeSampleSize int
	RelationType       RelationType
	Smooth             float64
	SubsampleThreshold float64
	ToLower            bool
	UpdateLRBatch      int
	Verbose            bool
	Window             int
}

func DefaultOptions() Options {
	return Options{
		BatchSize:          defaultBatchSize,
		Dim:                defaultDim,
		DocInMemory:        defaultDocInMemory,
		Goroutines:         defaultGoroutines,
		Initlr:             defaultInitlr,
		Iter:               defaultIter,
		LogBatch:           defaultLogBatch,
		MaxCount:           defaultMaxCount,
		MinCount:           defaultMinCount,
		MinLR:              defaultMinLR,
		NegativeSampleSize: defaultNegativeSampleSize,
		RelationType:       defaultRelationType,
		Smooth:             defaultSmooth,
		SubsampleThreshold: defaultSubsampleThreshold,
		ToLower:            defaultToLower,
		UpdateLRBatch:      defaultUpdateLRBatch,
		Verbose:            defaultVerbose,
		Window:             defaultWindow,
	}
}
func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().IntVar(&opts.BatchSize, "batch", defaultBatchSize, "batch size to train")
	cmd.Flags().IntVarP(&opts.Dim, "dim", "d", defaultDim, "dimension for word vector")
	cmd.Flags().IntVar(&opts.Goroutines, "goroutines", defaultGoroutines, "number of goroutine")
	cmd.Flags().BoolVar(&opts.DocInMemory, "in-memory", defaultDocInMemory, "whether to store the doc in memory")
	cmd.Flags().Float64Var(&opts.Initlr, "initlr", defaultInitlr, "initial learning rate")
	cmd.Flags().IntVar(&opts.Iter, "iter", defaultIter, "number of iteration")
	cmd.Flags().IntVar(&opts.LogBatch, "log-batch", defaultLogBatch, "batch size to log for counting words")
	cmd.Flags().IntVar(&opts.MaxCount, "max-count", defaultMaxCount, "upper limit to filter words")
	cmd.Flags().IntVar(&opts.MinCount, "min-count", defaultMinCount, "lower limit to filter words")
	cmd.Flags().Float64Var(&opts.MinLR, "min-lr", defaultMinLR, "lower limit of learning rate")
	cmd.Flags().IntVar(&opts.NegativeSampleSize, "sample", defaultNegativeSampleSize, "negative sample size")
	cmd.Flags().StringVar(&opts.RelationType, "rel", defaultRelationType, fmt.Sprintf("relation type for co-occurrence words. One of %s|%s|%s|%s", PPMI, PMI, Collocation, LogCollocation))
	cmd.Flags().Float64Var(&opts.Smooth, "smooth", defaultSmooth, "smoothing value for co-occurence value")
	cmd.Flags().Float64Var(&opts.SubsampleThreshold, "threshold", defaultSubsampleThreshold, "threshold for subsampling")
	cmd.Flags().BoolVar(&opts.ToLower, "to-lower", defaultToLower, "whether the words on corpus convert to lowercase or not")
	cmd.Flags().IntVar(&opts.UpdateLRBatch, "update-lr-batch", defaultUpdateLRBatch, "batch size to update learning rate")
	cmd.Flags().BoolVar(&opts.Verbose, "verbose", defaultVerbose, "verbose mode")
	cmd.Flags().IntVarP(&opts.Window, "window", "w", defaultWindow, "context window size")

}

type ModelOption func(*Options)

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

func LogBatch(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.LogBatch = v
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

func MinLR(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.MinLR = v
	})
}

func NegativeSampleSize(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.NegativeSampleSize = v
	})
}

func Relation(typ RelationType) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.RelationType = typ
	})
}

func Smooth(v float64) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.Smooth = v
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

func UpdateLRBatch(v int) ModelOption {
	return ModelOption(func(opts *Options) {
		opts.UpdateLRBatch = v
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
