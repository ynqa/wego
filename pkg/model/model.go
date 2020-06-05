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

package model

import (
	"io"
	"runtime"

	"github.com/spf13/cobra"
	"github.com/ynqa/wego/pkg/model/modelutil/save"
)

type Model interface {
	Train(io.Reader) error
	Save(io.Writer, save.VectorType) error
}

var (
	defaultBatchSize  = 100000
	defaultDim        = 10
	defaultInitlr     = 0.025
	defaultIter       = 15
	defaultThreadSize = runtime.NumCPU()
	defaultWindow     = 5
	defaultVerbose    = false
)

// Options stores common options for each model.
type Options struct {
	BatchSize  int
	Dim        int
	Initlr     float64
	Iter       int
	ThreadSize int
	Window     int
	Verbose    bool
}

func DefaultOptions() Options {
	return Options{
		BatchSize:  defaultBatchSize,
		Dim:        defaultDim,
		Initlr:     defaultInitlr,
		Iter:       defaultIter,
		ThreadSize: defaultThreadSize,
		Window:     defaultWindow,
		Verbose:    defaultVerbose,
	}
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().IntVar(&opts.BatchSize, "batch", defaultBatchSize, "batch size to train")
	cmd.Flags().IntVarP(&opts.Dim, "dim", "d", defaultDim, "dimension for word vector")
	cmd.Flags().Float64Var(&opts.Initlr, "initlr", defaultInitlr, "initial learning rate")
	cmd.Flags().IntVar(&opts.Iter, "iter", defaultIter, "number of iteration")
	cmd.Flags().IntVar(&opts.ThreadSize, "thread", defaultThreadSize, "number of goroutine")
	cmd.Flags().IntVarP(&opts.Window, "window", "w", defaultWindow, "context window size")
	cmd.Flags().BoolVar(&opts.Verbose, "verbose", defaultVerbose, "verbose mode")
}
