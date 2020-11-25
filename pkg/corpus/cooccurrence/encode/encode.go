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

package encode

// data structure for co-occurrence mapping:
// - https://blog.chewxy.com/2017/07/12/21-bits-english/

// EncodeBigram creates id between two words.
func EncodeBigram(l1, l2 uint64) uint64 {
	if l1 < l2 {
		return encode(l1, l2)
	} else {
		return encode(l2, l1)
	}
}

func encode(l1, l2 uint64) uint64 {
	return l1 | (l2 << 32)
}

// DecodeBigram reverts pair id to two word ids.
func DecodeBigram(pid uint64) (uint64, uint64) {
	f := pid >> 32
	return pid - (f << 32), f
}
