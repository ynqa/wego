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

var NULL = struct{}{}

type String map[string]struct{}

func New(v ...string) String {
	ss := String{}
	ss.Add(v...)
	return ss
}

func (s String) Add(v ...string) {
	for _, e := range v {
		s[e] = NULL
	}
}

func (s String) Contain(v string) bool {
	_, existed := (s)[v]
	return existed
}

func (s String) Len() int {
	return len(s)
}

func (s1 String) Equal(s2 String) bool {
	f := func() bool {
		for item := range s2 {
			if !s1.Contain(item) {
				return false
			}
		}
		return true
	}
	return len(s1) == len(s2) && f()
}
