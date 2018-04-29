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

// Word2vecConfig is enum of the Word2vec config.
type Word2vecConfig int

// The list of Word2vecConfig.
const (
	Model Word2vecConfig = iota
	Optimizer
	BatchSize
	MaxDepth
	NegativeSampleSize
	SubsampleThreshold
	Theta
)

// The defaults of Word2vecConfig.
const (
	DefaultModel              string  = "cbow"
	DefaultOptimizer          string  = "hs"
	DefaultBatchSize          int     = 10000
	DefaultMaxDepth           int     = 0
	DefaultNegativeSampleSize int     = 5
	DefaultSubsampleThreshold float64 = 1.0e-3
	DefaultTheta              float64 = 1.0e-4
)

func (w Word2vecConfig) String() string {
	switch w {
	case Model:
		return "model"
	case Optimizer:
		return "optimizer"
	case BatchSize:
		return "batchSize"
	case MaxDepth:
		return "maxDepth"
	case NegativeSampleSize:
		return "sample"
	case SubsampleThreshold:
		return "threshold"
	case Theta:
		return "theta"
	default:
		return "unknown"
	}
}
