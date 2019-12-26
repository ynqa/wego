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

	"github.com/ynqa/wego/pkg/config"
)

// RootCmd is the root command for word embedding.
var RootCmd = &cobra.Command{
	Use:   "wego",
	Short: "tools for embedding words into vector space",
	RunE: func(cmd *cobra.Command, args []string) error {
		return errors.Errorf("Set sub-command. One of %s|%s|%s|%s|%s",
			word2vecCmd.Name(), gloveCmd.Name(), lexvecCmd.Name(), searchCmd.Name(), replCmd.Name())
	},
}

func configFlagSet() *pflag.FlagSet {
	fs := pflag.NewFlagSet(RootCmd.Name(), pflag.ExitOnError)
	fs.StringP(config.InputFile.String(), "i", config.DefaultInputFile,
		"input file path for corpus")
	fs.StringP(config.OutputFile.String(), "o", config.DefaultOutputFile,
		"output file path to save word vectors")
	fs.IntP(config.Dimension.String(), "d", config.DefaultDimension,
		"dimension of word vector")
	fs.Int(config.Iteration.String(), config.DefaultIteration,
		"number of iteration")
	fs.Int(config.MinCount.String(), config.DefaultMinCount,
		"lower limit to filter rare words")
	fs.Int(config.ThreadSize.String(), config.DefaultThreadSize,
		"number of goroutine")
	fs.Int(config.BatchSize.String(), config.DefaultBatchSize,
		"interval word size to preprocess/train")
	fs.IntP(config.Window.String(), "w", config.DefaultWindow,
		"context window size")
	fs.Float64(config.Initlr.String(), config.DefaultInitlr,
		"initial learning rate")
	fs.Bool(config.Prof.String(), config.DefaultProf,
		"profiling mode to check the performances")
	fs.Bool(config.ToLower.String(), config.DefaultToLower,
		"whether the words on corpus convert to lowercase or not")
	fs.Bool(config.Verbose.String(), config.DefaultVerbose,
		"verbose mode")
	fs.String(config.SaveVectorType.String(), config.DefaultSaveVectorType.String(),
		"save vector type. One of: normal|add")
	return fs
}

func bindConfig(cmd *cobra.Command) {
	viper.BindPFlag(config.InputFile.String(), cmd.Flags().Lookup(config.InputFile.String()))
	viper.BindPFlag(config.OutputFile.String(), cmd.Flags().Lookup(config.OutputFile.String()))
	viper.BindPFlag(config.Dimension.String(), cmd.Flags().Lookup(config.Dimension.String()))
	viper.BindPFlag(config.Iteration.String(), cmd.Flags().Lookup(config.Iteration.String()))
	viper.BindPFlag(config.MinCount.String(), cmd.Flags().Lookup(config.MinCount.String()))
	viper.BindPFlag(config.ThreadSize.String(), cmd.Flags().Lookup(config.ThreadSize.String()))
	viper.BindPFlag(config.BatchSize.String(), cmd.Flags().Lookup(config.BatchSize.String()))
	viper.BindPFlag(config.Window.String(), cmd.Flags().Lookup(config.Window.String()))
	viper.BindPFlag(config.Initlr.String(), cmd.Flags().Lookup(config.Initlr.String()))
	viper.BindPFlag(config.Prof.String(), cmd.Flags().Lookup(config.Prof.String()))
	viper.BindPFlag(config.ToLower.String(), cmd.Flags().Lookup(config.ToLower.String()))
	viper.BindPFlag(config.Verbose.String(), cmd.Flags().Lookup(config.Verbose.String()))
	viper.BindPFlag(config.SaveVectorType.String(), cmd.Flags().Lookup(config.SaveVectorType.String()))
}

func init() {
	RootCmd.AddCommand(word2vecCmd)
	RootCmd.AddCommand(gloveCmd)
	RootCmd.AddCommand(lexvecCmd)
	RootCmd.AddCommand(searchCmd)
	RootCmd.AddCommand(replCmd)
}
