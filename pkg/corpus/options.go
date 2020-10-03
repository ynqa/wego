package corpus

import (
	"github.com/spf13/cobra"
)

const (
	defaultDocInMemory = false
	defaultToLower     = false
)

type Options struct {
	DocInMemory bool
	ToLower     bool
}

func DefaultOptions() Options {
	return Options{
		DocInMemory: defaultDocInMemory,
		ToLower:     defaultToLower,
	}
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().BoolVar(&opts.DocInMemory, "in-memory", defaultDocInMemory, "whether to store the doc in memory")
	cmd.Flags().BoolVar(&opts.ToLower, "lower", defaultToLower, "whether the words on corpus convert to lowercase or not")
}
