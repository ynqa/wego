// Copyright © 2019 Makoto Ito
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

	"github.com/ynqa/wego/pkg/builder"
	"github.com/ynqa/wego/pkg/model/word2vec"
)

func main() {
	b := builder.NewWord2vecBuilder()

	b.Dimension(10).
		Window(5).
		Model(word2vec.CBOW).
		Optimizer(word2vec.NEGATIVE_SAMPLING).
		NegativeSampleSize(5).
		Verbose()

	m, err := b.Build()
	if err != nil {
		// Failed to build word2vec.
	}

	input, _ := os.Open("text8")

	// Start to Train.
	if err = m.Train(input); err != nil {
		// Failed to train by word2vec.
	}

	// Save word vectors to a text file.
	m.Save("example.txt")
}
