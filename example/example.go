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

package main

import (
	"os"

	"github.com/ynqa/word-embedding/builder"
)

func main() {
	b := builder.NewWord2VecBuilder()

	b.SetDimension(10).
		SetWindow(5).
		SetModel("cbow").
		SetOptimizer("ns").
		SetNegativeSampleSize(5).
		SetVerbose()

	m, err := b.Build()

	if err != nil {
		// Failed to build word2vec.
	}

	inputFile, _ := os.Open("text8")

	f, err := m.Preprocess(inputFile)

	if err != nil {
		// Failed to Preprocess.
	}

	// Start to Train.
	m.Train(f)

	// Save word vectors to a text file.
	m.Save("example.txt")
}
