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

// Config stores the common config.
type Config struct {
	Dimension        int
	Iteration        int
	MinCount         int
	Thread           int
	Window           int
	InitLearningRate float64
	ToLower          bool
	Verbose          bool
}

// NewConfig creates *Config
func NewConfig(dimension, iteration, minCount, thread, window int, initlr float64, toLower, verbose bool) *Config {
	return &Config{
		Dimension:        dimension,
		Iteration:        iteration,
		MinCount:         minCount,
		Thread:           thread,
		Window:           window,
		InitLearningRate: initlr,
		ToLower:          toLower,
		Verbose:          verbose,
	}
}
