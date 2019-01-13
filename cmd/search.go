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

	"github.com/ynqa/wego/config"
	"github.com/ynqa/wego/search"
)

// SearchCmd is the subcommand to estimate similarity.
var SearchCmd = &cobra.Command{
	Use:     "search",
	Short:   "Search similar words",
	Long:    "Search similar words",
	Example: "  wego search -i example/word_vectors.txt microsoft",
	PreRun: func(cmd *cobra.Command, args []string) {
		searchBind(cmd)
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) == 1 {
			return executeSearch(args[0])
		}
		return errors.New("Input a single word")
	},
}

func init() {
	SearchCmd.Flags().StringP(config.InputFile.String(), "i", config.DefaultInputFile,
		"input file path for trained word vector")
	SearchCmd.Flags().IntP(config.Rank.String(), "r", config.DefaultRank,
		"how many the most similar words will be displayed")
}

func searchBind(cmd *cobra.Command) {
	viper.BindPFlag(config.Rank.String(), cmd.Flags().Lookup(config.Rank.String()))
	viper.BindPFlag(config.InputFile.String(), cmd.Flags().Lookup(config.InputFile.String()))
}

func executeSearch(target string) error {
	inputFile := viper.GetString(config.InputFile.String())
	rank := viper.GetInt(config.Rank.String())
	searcher := search.NewSearcher(target, rank)

	f, err := os.Open(inputFile)
	if err != nil {
		return err
	}

	if err := searcher.Search(f); err != nil {
		return err
	}

	return searcher.Describe()
}
