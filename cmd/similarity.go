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
	"fmt"

	"github.com/spf13/cobra"
	sim "github.com/ynqa/word-embedding/similarity"
	"github.com/ynqa/word-embedding/utils"
)

var (
	rank                 int
	inputVectorFile      string
	outputSimilarityFile string
)

// SimilarityCmd is the command for calculation of similarity.
var SimilarityCmd = &cobra.Command{
	Use:   "sim",
	Short: "Estimate the similarity between words",
	Long:  "Estimate the similarity between words",
	Run: func(cmd *cobra.Command, args []string) {
		if !inputVectorFileIsExist() {
			utils.Fatal(fmt.Errorf("InputFile %s is not existed", inputVectorFile))
		} else if outputSimilarityFileIsExist() {
			utils.Fatal(fmt.Errorf("OutputFile %s is already existed", outputSimilarityFile))
		}

		if len(args) == 1 {
			describe(args[0])
		} else {
			utils.Fatal(errors.New("Input a single word"))
		}
	},
}

func init() {
	SimilarityCmd.Flags().IntVarP(&rank, "rank", "r", 10, "Set number of the similar word list displayed")
	SimilarityCmd.Flags().StringVarP(&inputVectorFile, "input", "i", "example/word_vectors.txt",
		"Input path of a file written words' vector with libsvm format")
	SimilarityCmd.Flags().StringVarP(&outputSimilarityFile, "output", "o", "",
		"Output path of a file written similar word list")
}

func describe(w string) {
	if err := sim.Load(inputVectorFile); err != nil {
		utils.Fatal(err)
	}

	if err := sim.Describe(w, rank); err != nil {
		utils.Fatal(err)
	}
}
