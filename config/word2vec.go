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

// Word2VecConfig is enum of the Word2Vec config.
type Word2VecConfig int

// The list of Word2VecConfig.
const (
	Model Word2VecConfig = iota
	Optimizer
	MaxDepth
	NegativeSampleSize
	Theta
	BatchSize
	SubsampleThreshold
)

// The defaults of Word2VecConfig.
const (
	DefaultModel              string  = "cbow"
	DefaultOptimizer          string  = "hs"
	DefaultMaxDepth           int     = 0
	DefaultNegativeSampleSize int     = 5
	DefaultTheta              float64 = 1.0e-4
	DefaultBatchSize          int     = 10000
	DefaultSubsampleThreshold float64 = 1.0e-3
)

func (w Word2VecConfig) String() string {
	switch w {
	case Model:
		return "model"
	case Optimizer:
		return "optimizer"
	case MaxDepth:
		return "maxDepth"
	case NegativeSampleSize:
		return "sample"
	case Theta:
		return "theta"
	case BatchSize:
		return "batchSize"
	case SubsampleThreshold:
		return "threshold"
	default:
		return "unknown"
	}
}
