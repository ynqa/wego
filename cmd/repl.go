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

package cmd

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/ynqa/wego/pkg/config"
	"github.com/ynqa/wego/pkg/repl"
)

var replCmd = &cobra.Command{
	Use:   "repl",
	Short: "Search similar words with REPL mode",
	Long:  "Search similar words with REPL mode",
	Example: "  wego repl -i example/word_vectors.txt\n" +
		"  >> apple + banana\n" +
		"  ...",
	PreRun: func(cmd *cobra.Command, args []string) {
		replBind(cmd)
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		return executeRepl()
	},
}

func init() {
	replCmd.Flags().StringP(config.InputFile.String(), "i", config.DefaultOutputFile,
		"input file path for trained word vector")
	replCmd.Flags().IntP(config.Rank.String(), "r", config.DefaultRank,
		"how many the most similar words will be displayed")
}

func replBind(cmd *cobra.Command) {
	viper.BindPFlag(config.InputFile.String(), cmd.Flags().Lookup(config.InputFile.String()))
	viper.BindPFlag(config.Rank.String(), cmd.Flags().Lookup(config.Rank.String()))
}

func executeRepl() error {
	inputFile := viper.GetString(config.InputFile.String())
	f, err := os.Open(inputFile)
	if err != nil {
		return err
	}
	defer f.Close()

	k := viper.GetInt(config.Rank.String())
	repl, err := repl.NewRepl(f, k)
	if err != nil {
		return err
	}
	return repl.Run()
}
