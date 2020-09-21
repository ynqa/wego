// Copyright © 2020 wego authors
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

package save

import (
	"github.com/pkg/errors"
)

func InvalidVectorTypeError(typ VectorType) error {
	return errors.Errorf("invalid vector type: %s not in %s|%s", typ, Single, Aggregated)
}

type VectorType string

const (
	Single                VectorType = "single"
	Aggregated            VectorType = "agg"
	defaultSaveVectorType            = Single
)

func (t *VectorType) String() string {
	if *t == VectorType("") {
		*t = defaultSaveVectorType
	}
	return string(*t)
}

func (t *VectorType) Set(name string) error {
	typ := VectorType(name)
	if typ == Single || typ == Aggregated {
		*t = typ
		return nil
	}
	return InvalidVectorTypeError(typ)
}

func (t *VectorType) Type() string {
	return t.String()
}
