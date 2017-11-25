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

// GloveCmd is for Glove command.
var GloveCmd = &cobra.Command{
	Use:   "glove",
	Short: "Embed words using glove",
	Long:  "Embed words using glove",
	PreRun: func(cmd *cobra.Command, args []string) {
		configBind(cmd)
		gloveBind(cmd)
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

		return runGlove()
	},
}

func init() {
	GloveCmd.Flags().AddFlagSet(ConfigFlagSet())
	GloveCmd.Flags().BoolP("help", "h", false, "Help for "+GloveCmd.Name())
	GloveCmd.Flags().String(config.Solver.String(), config.DefaultSolver,
		"Set the solver of GloVe. One of: sgd|adagrad")
	GloveCmd.Flags().Int(config.Iteration.String(), config.DefaultIteration,
		"Set the iteration")
	GloveCmd.Flags().Float64(config.Alpha.String(), config.DefaultAlpha,
		"Set alpha")
	GloveCmd.Flags().Int(config.Xmax.String(), config.DefaultXmax,
		"Set xmax")
	GloveCmd.Flags().Int(config.MinCount.String(), config.DefaultMinCount,
		"Set the min count to filter rare words")
	GloveCmd.Flags().Int(config.BatchSize.String(), config.DefaultBatchSize,
		"Set the batch size to buffer words referred as chunk")
}

func gloveBind(cmd *cobra.Command) {
	viper.BindPFlag(config.Solver.String(), cmd.Flags().Lookup(config.Solver.String()))
	viper.BindPFlag(config.Iteration.String(), cmd.Flags().Lookup(config.Iteration.String()))
	viper.BindPFlag(config.Alpha.String(), cmd.Flags().Lookup(config.Alpha.String()))
	viper.BindPFlag(config.Xmax.String(), cmd.Flags().Lookup(config.Xmax.String()))
	viper.BindPFlag(config.MinCount.String(), cmd.Flags().Lookup(config.MinCount.String()))
	viper.BindPFlag(config.BatchSize.String(), cmd.Flags().Lookup(config.BatchSize.String()))
}

func runGlove() error {
	inputFile := viper.GetString(config.InputFile.String())

	if !validate.FileExists(inputFile) {
		return errors.Errorf("Not such a file %s", inputFile)
	}

	outputFile := viper.GetString(config.OutputFile.String())

	if validate.FileExists(outputFile) {
		return errors.Errorf("%s is already existed", outputFile)
	}

	glove := builder.NewGloveBuilderViper()

	mod, err := glove.Build()
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
