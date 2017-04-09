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

package set

var empty = struct{}{}

// String stores string elements in native map.
type String map[string]struct{}

// New creates a set with items.
func New(items ...string) String {
	ss := String{}
	ss.Add(items...)
	return ss
}

// Add adds the items (one or more) to the set.
func (s String) Add(items ...string) {
	for _, i := range items {
		s[i] = empty
	}
}

// Contain checks whether an item is present in the set or not.
func (s String) Contain(item string) bool {
	_, existed := (s)[item]
	return existed
}

// Len returns the number of elements in the set.
func (s String) Len() int {
	return len(s)
}

// Equal checks whether input set is equal to it or not.
func (s String) Equal(ss String) bool {
	f := func() bool {
		for item := range ss {
			if !s.Contain(item) {
				return false
			}
		}
		return true
	}
	return len(s) == len(ss) && f()
}
