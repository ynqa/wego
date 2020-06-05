package corpus

import (
	"github.com/spf13/cobra"
)

const (
	defaultMinCount = 5
	defaultToLower  = false
)

type Options struct {
	MinCount int
	ToLower  bool
}

func DefaultOptions() Options {
	return Options{
		MinCount: defaultMinCount,
		ToLower:  defaultToLower,
	}
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().IntVar(&opts.MinCount, "min-count", defaultMinCount, "lower limit to filter rare words")
	cmd.Flags().BoolVar(&opts.ToLower, "lower", defaultToLower, "whether the words on corpus convert to lowercase or not")
}
