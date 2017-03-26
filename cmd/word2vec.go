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

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/ynqa/word-embedding/models/word2vec"
	"github.com/ynqa/word-embedding/models/word2vec/model"
	"github.com/ynqa/word-embedding/models/word2vec/opt"
	"github.com/ynqa/word-embedding/utils"
)

var (
	subModel, optimizer  string
	maxDepth, sampleSize int
)

var Word2vecCmd = &cobra.Command{
	Use:   "word2vec",
	Short: "Embed words using word2vec",
	Long:  "Embed words using word2vec",
	Run: func(cmd *cobra.Command, args []string) {
		if !inputFileIsExist() {
			utils.Fatal(fmt.Errorf("InputFile %s is not existed", inputFile))
		} else if outputFileIsExist() {
			utils.Fatal(fmt.Errorf("OutputFile %s is already existed", outputFile))
		}

		if err := validateCommonParams(); err != nil {
			utils.Fatal(err)
		} else if err := validateWord2vecParams(); err != nil {
			utils.Fatal(err)
		}

		start()
	},
}

func init() {
	Word2vecCmd.Flags().AddFlagSet(GetCommonFlagSet())
	Word2vecCmd.Flags().StringVar(&subModel, "model", "cbow", "Set model from: skip-gram|cbow")
	Word2vecCmd.Flags().StringVar(&optimizer, "optimizer", "hs", "Set optimizer from: hs|ns")
	Word2vecCmd.Flags().IntVar(&maxDepth, "max-depth", 0, "Set number of times to track huffman tree, "+
		"max-depth=0 means tracking full path (using only hierarchical softmax)")
	Word2vecCmd.Flags().IntVar(&sampleSize, "negative", 5, "Set number of negative samplings (using only negative sampling)")
}

func NewWord2Vec() word2vec.Word2Vec {
	return word2vec.Word2Vec{
		Common: NewCommon(),
		Model:  NewModel(),
		Opt:    NewOptimizer(),
	}
}

func NewOptimizer() (o word2vec.Optimizer) {
	switch optimizer {
	case "hs":
		o = opt.HierarchicalSoftmax{
			Common:   NewCommon(),
			MaxDepth: maxDepth,
		}
	case "ns":
		o = opt.NegativeSampling{
			Common:     NewCommon(),
			SampleSize: sampleSize,
		}
	}
	return
}

func NewModel() (m word2vec.Model) {
	switch subModel {
	case "skip-gram":
		m = model.SkipGram{
			Common: NewCommon(),
		}
	case "cbow":
		m = model.CBOW{
			Common: NewCommon(),
		}
	}
	return
}

func start() {
	w2v := NewWord2Vec()

	fmt.Print("Start PreTrain...\n")
	if err := w2v.PreTrain(); err != nil {
		utils.Fatal(err)
	}
	fmt.Print("Finish PreTrain\n")

	fmt.Printf("Number of terms: %d\n", word2vec.GetTerms())
	fmt.Printf("Number of words: %d\n", word2vec.GetWords())

	fmt.Print("Start Train...\n")
	w2v.Run()
	fmt.Print("Finish Train\n")

	if err := w2v.Save(); err != nil {
		utils.Fatal(err)
	}
	fmt.Print("Finish Save!\n")
}
