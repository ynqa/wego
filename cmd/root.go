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
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	"github.com/ynqa/word-embedding/config"
)

// RootCmd is the root command for word embedding.
var RootCmd = &cobra.Command{
	Use:   "word-embedding",
	Short: "The tools embedding words into vector space",
	Long:  "The tools embedding words into vector space",
	RunE: func(cmd *cobra.Command, args []string) error {
		return errors.New("Set sub-command from: distance|word2vec")
	},
}

// ConfigFlagSet creates the common config flags.
func ConfigFlagSet() *pflag.FlagSet {
	fs := pflag.NewFlagSet(RootCmd.Name(), pflag.ExitOnError)
	fs.StringP(config.InputFile.String(), "i", config.DefaultInputFile,
		"Set the input file path to load corpus")
	fs.StringP(config.OutputFile.String(), "o", config.DefaultOutputFile,
		"Set the output file path to save word vectors")
	fs.IntP(config.Dimension.String(), "d", config.DefaultDimension,
		"Set the dimension of word vector")
	fs.IntP(config.Window.String(), "w", config.DefaultWindow,
		"Set the context window size")
	fs.Int(config.Thread.String(), config.DefaultThread,
		"Set number of parallel")
	fs.Float64(config.InitLearningRate.String(), config.DefaultInitLearningRate,
		"Set the initial learning rate")
	fs.Bool(config.Prof.String(), config.DefaultProf,
		"Profiling mode to check the performances")
	fs.Bool(config.ToLower.String(), config.DefaultToLower,
		"Whether the words on corpus convert to lowercase or not")
	fs.Bool(config.Verbose.String(), config.DefaultVerbose,
		"Verbose mode")
	return fs
}

func configBind(cmd *cobra.Command) {
	viper.BindPFlag(config.InputFile.String(), cmd.Flags().Lookup(config.InputFile.String()))
	viper.BindPFlag(config.OutputFile.String(), cmd.Flags().Lookup(config.OutputFile.String()))
	viper.BindPFlag(config.Dimension.String(), cmd.Flags().Lookup(config.Dimension.String()))
	viper.BindPFlag(config.Window.String(), cmd.Flags().Lookup(config.Window.String()))
	viper.BindPFlag(config.InitLearningRate.String(), cmd.Flags().Lookup(config.InitLearningRate.String()))
	viper.BindPFlag(config.Thread.String(), cmd.Flags().Lookup(config.Thread.String()))
	viper.BindPFlag(config.Prof.String(), cmd.Flags().Lookup(config.Prof.String()))
	viper.BindPFlag(config.ToLower.String(), cmd.Flags().Lookup(config.ToLower.String()))
	viper.BindPFlag(config.Verbose.String(), cmd.Flags().Lookup(config.Verbose.String()))
}

func init() {
	RootCmd.Flags().BoolP("help", "h", false, "Help for "+RootCmd.Name())
	RootCmd.AddCommand(Word2VecCmd)
	RootCmd.AddCommand(DistanceCmd)
}
