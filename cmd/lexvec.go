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
	"runtime/pprof"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/ynqa/wego/pkg/builder"
	"github.com/ynqa/wego/pkg/config"
	"github.com/ynqa/wego/pkg/validate"
)

var lexvecCmd = &cobra.Command{
	Use:   "lexvec",
	Short: "Lexvec: Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations",
	PreRun: func(cmd *cobra.Command, args []string) {
		bindConfig(cmd)
		bindLexvec(cmd)
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
		return runLexvec()
	},
}

func init() {
	lexvecCmd.Flags().AddFlagSet(configFlagSet())
	lexvecCmd.Flags().Int(config.NegativeSampleSize.String(), config.DefaultNegativeSampleSize,
		"negative sample size(for negative sampling only)")
	lexvecCmd.Flags().Float64(config.Theta.String(), config.DefaultTheta,
		"lower limit of learning rate (lr >= initlr * theta)")
	lexvecCmd.Flags().Float64(config.Smooth.String(), config.DefaultSmooth,
		"smoothing value for co-occurence value")
	lexvecCmd.Flags().String(config.RelationType.String(), config.DefaultRelationType.String(),
		"relation type for counting co-occurrence. One of ppmi|pmi|co|logco")
}

func bindLexvec(cmd *cobra.Command) {
	viper.BindPFlag(config.NegativeSampleSize.String(), cmd.Flags().Lookup(config.NegativeSampleSize.String()))
	viper.BindPFlag(config.Theta.String(), cmd.Flags().Lookup(config.Theta.String()))
	viper.BindPFlag(config.Smooth.String(), cmd.Flags().Lookup(config.Smooth.String()))
	viper.BindPFlag(config.RelationType.String(), cmd.Flags().Lookup(config.RelationType.String()))
}

func runLexvec() error {
	outputFile := viper.GetString(config.OutputFile.String())
	if validate.FileExists(outputFile) {
		return errors.Errorf("%s is already existed", outputFile)
	}
	lexvec, err := builder.NewLexvecBuilderFromViper()
	if err != nil {
		return err
	}
	mod, err := lexvec.Build()
	if err != nil {
		return err
	}
	inputFile := viper.GetString(config.InputFile.String())
	if !validate.FileExists(inputFile) {
		return errors.Errorf("Not such a file %s", inputFile)
	}
	input, err := os.Open(inputFile)
	if err != nil {
		return err
	}
	if err := mod.Train(input); err != nil {
		return err
	}
	return mod.Save(outputFile)
}
