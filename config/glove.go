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

// GloveConfig is enum of the GloVe config.
type GloveConfig int

// The list of GloveConfig.
const (
	Solver GloveConfig = iota
	Xmax
	Alpha
)

// The defaults of GloveConfig.
const (
	DefaultSolver string  = "sgd"
	DefaultXmax   int     = 100
	DefaultAlpha  float64 = 0.75
)

func (g GloveConfig) String() string {
	switch g {
	case Solver:
		return "solver"
	case Xmax:
		return "xmax"
	case Alpha:
		return "alpha"
	default:
		return "unknown"
	}
}
