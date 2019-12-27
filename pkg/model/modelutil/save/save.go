package save

import (
	"github.com/pkg/errors"
)

func InvalidVectorTypeError(typ VectorType) error {
	return errors.Errorf("invalid vector type: %s not in %s|%s", typ, SingleVector, AggregatedVector)
}

type VectorType string

const (
	SingleVector          VectorType = "single"
	AggregatedVector      VectorType = "agg"
	defaultSaveVectorType            = SingleVector
)

func (t *VectorType) String() string {
	if *t == VectorType("") {
		*t = defaultSaveVectorType
	}
	return string(*t)
}

func (t *VectorType) Set(name string) error {
	typ := VectorType(name)
	if typ == SingleVector || typ == AggregatedVector {
		*t = typ
		return nil
	}
	return InvalidVectorTypeError(typ)
}

func (t *VectorType) Type() string {
	return t.String()
}
