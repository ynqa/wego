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

	"github.com/ynqa/wego/builder"
	"github.com/ynqa/wego/config"
	"github.com/ynqa/wego/validate"
)

// Word2vecCmd is the subcommand for Word2vec.
var Word2vecCmd = &cobra.Command{
	Use:   "word2vec",
	Short: "Word2Vec: Continuous Bag-of-Words and Skip-gram model",
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

		return executeWord2vec()
	},
}

func init() {
	Word2vecCmd.Flags().AddFlagSet(ConfigFlagSet())
	Word2vecCmd.Flags().String(config.Model.String(), config.DefaultModel,
		"which model does it use? one of: cbow|skip-gram")
	Word2vecCmd.Flags().String(config.Optimizer.String(), config.DefaultOptimizer,
		"which optimizer does it use? one of: hs|ns")
	Word2vecCmd.Flags().Int(config.BatchSize.String(), config.DefaultBatchSize,
		"interval word size to update learning rate")
	Word2vecCmd.Flags().Int(config.MaxDepth.String(), config.DefaultMaxDepth,
		"times to track huffman tree, max-depth=0 means to track full path from root to word (for hierarchical softmax only)")
	Word2vecCmd.Flags().Int(config.NegativeSampleSize.String(), config.DefaultNegativeSampleSize,
		"negative sample size(for negative sampling only)")
	Word2vecCmd.Flags().Float64(config.SubsampleThreshold.String(), config.DefaultSubsampleThreshold,
		"threshold for subsampling")
	Word2vecCmd.Flags().Float64(config.Theta.String(), config.DefaultTheta,
		"lower limit of learning rate (lr >= initlr * theta)")
}

func word2vecBind(cmd *cobra.Command) {
	viper.BindPFlag(config.Model.String(), cmd.Flags().Lookup(config.Model.String()))
	viper.BindPFlag(config.Optimizer.String(), cmd.Flags().Lookup(config.Optimizer.String()))
	viper.BindPFlag(config.BatchSize.String(), cmd.Flags().Lookup(config.BatchSize.String()))
	viper.BindPFlag(config.MaxDepth.String(), cmd.Flags().Lookup(config.MaxDepth.String()))
	viper.BindPFlag(config.NegativeSampleSize.String(), cmd.Flags().Lookup(config.NegativeSampleSize.String()))
	viper.BindPFlag(config.SubsampleThreshold.String(), cmd.Flags().Lookup(config.SubsampleThreshold.String()))
	viper.BindPFlag(config.Theta.String(), cmd.Flags().Lookup(config.Theta.String()))
}

func executeWord2vec() error {
	outputFile := viper.GetString(config.OutputFile.String())
	if validate.FileExists(outputFile) {
		return errors.Errorf("%s is already existed", outputFile)
	}

	w2v := builder.NewWord2vecBuilderFromViper()
	mod, err := w2v.Build()
	if err != nil {
		return err
	}
	if err := mod.Train(); err != nil {
		return err
	}
	return mod.Save(outputFile)
}
