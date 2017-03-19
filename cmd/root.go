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
		utils.Fatal(errors.New("Set sub-command from: word2vec"))
	},
}

func init() {
	RootCmd.AddCommand(Word2vecCmd)
	RootCmd.PersistentFlags().StringVarP(&inputFile, "input", "i", "example/input.txt", "Input file path for learning")
	RootCmd.PersistentFlags().StringVarP(&outputFile, "output", "o", "example/output.txt", "Output file path for each learned word vector")
	RootCmd.PersistentFlags().IntVarP(&dimension, "dimension", "d", 10, "Set word vector dimension size")
	RootCmd.PersistentFlags().IntVarP(&window, "window", "w", 5, "Set window size")
	RootCmd.PersistentFlags().Float64Var(&learningRate, "lr", 0.25, "Set init learning rate")
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
