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
	"errors"

	"github.com/spf13/cobra"

	"github.com/ynqa/word-embedding/models"
	"github.com/ynqa/word-embedding/utils"
	flag "github.com/spf13/pflag"
)

var (
	inputFile, outputFile string
	dimension, window     int
	learningRate          float64
)

var RootCmd = &cobra.Command{
	Use:   "word-embedding",
	Short: "The tools embedding words into vector space",
	Long:  "The tools embedding words into vector space",
	Run: func(cmd *cobra.Command, args []string) {
		utils.Fatal(errors.New("Set sub-command from: distance|word2vec"))
	},
}

func GetCommonFlagSet() *flag.FlagSet {
	fs := flag.NewFlagSet(RootCmd.Name(), flag.ContinueOnError)
	fs.StringVarP(&inputFile, "input", "i", "example/input.txt", "Input file path for learning")
	fs.StringVarP(&outputFile, "output", "o", "example/word_vectors.txt", "Output file path for each learned word vector")
	fs.IntVarP(&dimension, "dimension", "d", 10, "Set word vector dimension size")
	fs.IntVarP(&window, "window", "w", 5, "Set window size")
	fs.Float64Var(&learningRate, "lr", 0.025, "Set init learning rate")
	return fs
}

func init() {
	RootCmd.AddCommand(Word2vecCmd)
}

func NewCommon() models.Common {
	return models.Common{
		InputFile:    inputFile,
		OutputFile:   outputFile,
		Dimension:    dimension,
		Window:       window,
		LearningRate: learningRate,
	}
}
