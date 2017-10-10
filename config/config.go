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

// Config is enum of the common config.
type Config int

// The list of Config.
const (
	InputFile Config = iota
	OutputFile
	Dimension
	Window
	InitLearningRate
	Dtype
	Prof
	ToLower
	Verbose
)

// The defaults of Config.
const (
	DefaultInputFile        string  = "example/input.txt"
	DefaultOutputFile       string  = "example/word_vectors.txt"
	DefaultDimension        int     = 10
	DefaultWindow           int     = 5
	DefaultInitLearningRate float64 = 0.025
	DefaultDtype            string  = "float64"
	DefaultProf             bool    = false
	DefaultToLower          bool    = false
	DefaultVerbose          bool    = false
)

func (c Config) String() string {
	switch c {
	case InputFile:
		return "inputFile"
	case OutputFile:
		return "outputFile"
	case Dimension:
		return "dimension"
	case Window:
		return "window"
	case InitLearningRate:
		return "initlr"
	case Dtype:
		return "dtype"
	case Prof:
		return "prof"
	case ToLower:
		return "lower"
	case Verbose:
		return "verbose"
	default:
		return "unknown"
	}
}
