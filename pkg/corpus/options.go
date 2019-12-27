package corpus

import (
	"github.com/spf13/cobra"
)

const (
	defaultToLower = false
)

type Options struct {
	ToLower bool
}

func DefaultOptions() Options {
	return Options{
		ToLower: defaultToLower,
	}
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().BoolVar(&opts.ToLower, "lower", defaultToLower, "whether the words on corpus convert to lowercase or not")
}
