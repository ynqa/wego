package pairwise

import (
	"fmt"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
)

func invalidCountTypeError(typ CountType) error {
	return errors.Errorf("invalid relation type: %s not in %s|%s", typ, Increment, Distance)
}

type CountType string

const (
	Increment        CountType = "inc"
	Distance         CountType = "dis"
	defaultCountType           = Increment
	defaultWindow              = 5
)

func (t *CountType) String() string {
	if *t == CountType("") {
		*t = defaultCountType
	}
	return string(*t)
}

func (t *CountType) Set(name string) error {
	typ := CountType(name)
	if typ == Increment || typ == Distance {
		*t = typ
		return nil
	}
	return invalidCountTypeError(typ)
}

func (t *CountType) Type() string {
	return t.String()
}

type Options struct {
	CountType CountType
	Window    int
}

func DefaultOptions() Options {
	return Options{
		CountType: defaultCountType,
		Window:    defaultWindow,
	}
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().Var(&opts.CountType, "cnt", fmt.Sprintf("count type for co-occurrence words. One of %s|%s", Increment, Distance))
}
