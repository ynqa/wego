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
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/ynqa/word-embedding/config"
	"github.com/ynqa/word-embedding/similarity"
)

// SimilarityCmd is the command for calculation of similarity.
var SimilarityCmd = &cobra.Command{
	Use:     "sim -i FILENAME WORD",
	Short:   "Estimate the similarity between words",
	Long:    "Estimate the similarity between words",
	Example: "  word-embedding sim -i example/word_vectors.txt microsoft",
	PreRun: func(cmd *cobra.Command, args []string) {
		similarityBind(cmd)
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) == 1 {
			return runSimilarity(args[0])
		}
		return errors.New("Input a single word")
	},
}

func init() {
	SimilarityCmd.Flags().BoolP("help", "h", false, "Help for "+SimilarityCmd.Name())
	SimilarityCmd.Flags().StringP(config.InputFile.String(), "i", config.DefaultInputFile,
		"Set the input file path to load word vector list")
	SimilarityCmd.Flags().IntP(config.Rank.String(), "r", config.DefaultRank,
		"How many the most similar words will be displayed")
}

func similarityBind(cmd *cobra.Command) {
	viper.BindPFlag(config.Rank.String(), cmd.Flags().Lookup(config.Rank.String()))
	viper.BindPFlag(config.InputFile.String(), cmd.Flags().Lookup(config.InputFile.String()))
}

func runSimilarity(target string) error {
	inputFile := viper.GetString(config.InputFile.String())
	rank := viper.GetInt(config.Rank.String())

	est := similarity.NewEstimator(target, rank)

	f, err := os.Open(inputFile)
	if err != nil {
		return err
	}

	if err := est.Estimate(f); err != nil {
		return err
	}

	return est.Describe()
}
