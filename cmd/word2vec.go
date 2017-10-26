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
	"runtime/pprof"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/ynqa/word-embedding/builder"
	"github.com/ynqa/word-embedding/config"
	"github.com/ynqa/word-embedding/validate"
)

// Word2VecCmd is the word2vec command.
var Word2VecCmd = &cobra.Command{
	Use:   "word2vec",
	Short: "Embed words using word2vec",
	Long:  "Embed words using word2vec",
	PreRun: func(cmd *cobra.Command, args []string) {
		configBind(cmd)
		word2vecBind(cmd)
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		if viper.GetBool(config.Prof.String()) {
			f, err := os.Create("cpu.prof")
			if err != nil {
				os.Exit(1)
			}
			pprof.StartCPUProfile(f)
			defer pprof.StopCPUProfile()
		}

		return runWord2Vec()
	},
}

func init() {
	Word2VecCmd.Flags().AddFlagSet(ConfigFlagSet())
	Word2VecCmd.Flags().BoolP("help", "h", false, "Help for "+Word2VecCmd.Name())
	Word2VecCmd.Flags().String(config.Model.String(), config.DefaultModel,
		"Set the model of Word2Vec. One of: cbow|skip-gram")
	Word2VecCmd.Flags().String(config.Optimizer.String(), config.DefaultOptimizer,
		"Set the optimizer of Word2Vec. One of: hs|ns")
	Word2VecCmd.Flags().Int(config.MaxDepth.String(), config.DefaultMaxDepth,
		"Set the number of times to track huffman tree, max-depth=0 means to track full path from root to word (using only hierarchical softmax)")
	Word2VecCmd.Flags().Int(config.NegativeSampleSize.String(), config.DefaultNegativeSampleSize,
		"Set the number of the samples as negative (using only negative sampling)")
	Word2VecCmd.Flags().Float64(config.Theta.String(), config.DefaultTheta,
		"Set the lower limit of learning rate (lr >= initlr * theta)")
	Word2VecCmd.Flags().Int(config.BatchSize.String(), config.DefaultBatchSize,
		"Set the batch size to update learning rate")
	Word2VecCmd.Flags().Float64(config.SubsampleThreshold.String(), config.DefaultSubsampleThreshold,
		"Set the threshold for subsampling")
}

func word2vecBind(cmd *cobra.Command) {
	viper.BindPFlag(config.Model.String(), cmd.Flags().Lookup(config.Model.String()))
	viper.BindPFlag(config.Optimizer.String(), cmd.Flags().Lookup(config.Optimizer.String()))
	viper.BindPFlag(config.MaxDepth.String(), cmd.Flags().Lookup(config.MaxDepth.String()))
	viper.BindPFlag(config.NegativeSampleSize.String(), cmd.Flags().Lookup(config.NegativeSampleSize.String()))
	viper.BindPFlag(config.BatchSize.String(), cmd.Flags().Lookup(config.BatchSize.String()))
	viper.BindPFlag(config.SubsampleThreshold.String(), cmd.Flags().Lookup(config.SubsampleThreshold.String()))
}

func runWord2Vec() error {
	inputFile := viper.GetString(config.InputFile.String())

	if !validate.FileExists(inputFile) {
		return errors.Errorf("Not such a file %s", inputFile)
	}

	outputFile := viper.GetString(config.OutputFile.String())

	if validate.FileExists(outputFile) {
		return errors.Errorf("%s is already existed", outputFile)
	}

	w2v := builder.NewWord2VecBuilderViper()

	mod, err := w2v.Build()
	if err != nil {
		return err
	}

	file, err := os.Open(inputFile)
	if err != nil {
		return err
	}

	f, err := mod.Preprocess(file)
	if err != nil {
		return err
	}

	if err := mod.Train(f); err != nil {
		return err
	}

	return mod.Save(outputFile)
}
