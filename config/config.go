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

package config

import (
	"runtime"

	"github.com/ynqa/wego/corpus"
	"github.com/ynqa/wego/model"
	"github.com/ynqa/wego/model/glove"
	"github.com/ynqa/wego/model/word2vec"
)

// Config is enum of the common config.
type Config int

// The list of Config.
const (
	InputFile Config = iota
	OutputFile
	Dimension
	Iteration
	MinCount
	ThreadSize
	BatchSize
	Window
	Initlr
	Prof
	ToLower
	Verbose
	SaveVectorType
	// Word2Vec
	Model
	Optimizer
	MaxDepth
	NegativeSampleSize
	SubsampleThreshold
	Theta
	// GloVe
	Solver
	Xmax
	Alpha
	// Lexvec
	RelationType
	Smooth
	// Search
	Rank
)

// The defaults of Config.
var (
	DefaultInputFile      string               = "example/input.txt"
	DefaultOutputFile     string               = "example/word_vectors.txt"
	DefaultDimension      int                  = 10
	DefaultIteration      int                  = 15
	DefaultMinCount       int                  = 5
	DefaultThreadSize     int                  = runtime.NumCPU()
	DefaultBatchSize      int                  = 10000
	DefaultWindow         int                  = 5
	DefaultInitlr         float64              = 0.025
	DefaultProf           bool                 = false
	DefaultToLower        bool                 = false
	DefaultVerbose        bool                 = false
	DefaultSaveVectorType model.SaveVectorType = model.NORMAL
	// Word2Vec
	DefaultModel              word2vec.ModelType     = word2vec.CBOW
	DefaultOptimizer          word2vec.OptimizerType = word2vec.NEGATIVE_SAMPLING
	DefaultMaxDepth           int                    = 0
	DefaultNegativeSampleSize int                    = 5
	DefaultSubsampleThreshold float64                = 1.0e-3
	DefaultTheta              float64                = 1.0e-4
	// GloVe
	DefaultSolver glove.SolverType = glove.SGD
	DefaultXmax   int              = 100
	DefaultAlpha  float64          = 0.75
	// Lexvex
	DefaultRelationType corpus.RelationType = corpus.PPMI
	DefaultSmooth       float64             = 0.75
	// Search
	DefaultRank int = 10
)

func (c Config) String() string {
	switch c {
	case InputFile:
		return "inputFile"
	case OutputFile:
		return "outputFile"
	case Dimension:
		return "dimension"
	case Iteration:
		return "iter"
	case MinCount:
		return "min-count"
	case ThreadSize:
		return "thread"
	case BatchSize:
		return "batchSize"
	case Window:
		return "window"
	case Initlr:
		return "initlr"
	case Prof:
		return "prof"
	case ToLower:
		return "lower"
	case Verbose:
		return "verbose"
	case SaveVectorType:
		return "save-vec"
	// Word2Vec
	case Model:
		return "model"
	case Optimizer:
		return "optimizer"
	case MaxDepth:
		return "maxDepth"
	case NegativeSampleSize:
		return "sample"
	case SubsampleThreshold:
		return "threshold"
	case Theta:
		return "theta"
	// GloVe
	case Solver:
		return "solver"
	case Xmax:
		return "xmax"
	case Alpha:
		return "alpha"
	// Lexvec
	case RelationType:
		return "rel"
	case Smooth:
		return "smooth"
	// Search
	case Rank:
		return "rank"
	default:
		return "unknown"
	}
}
