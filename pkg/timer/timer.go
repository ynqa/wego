// Copyright © 2019 Makoto Ito
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

package timer

import (
	"fmt"
	"time"
)

const (
	ms = 1000
	s  = 60
	m  = 60
	h  = 24
)

type Timer struct {
	start, last time.Time
}

func NewTimer() *Timer {
	n := time.Now()
	return &Timer{
		start: n,
		last:  n,
	}
}

func (t *Timer) AllElapsed() string {
	d := time.Now().Sub(t.start)
	return durationFmt(d)
}

func (t *Timer) Elapsed() string {
	n := time.Now()
	d := n.Sub(t.last)
	t.last = n
	return durationFmt(d)
}

func durationFmt(d time.Duration) string {
	second := int(d.Seconds()) % s
	minute := int(d.Minutes()) % m
	hour := int(d.Hours()) % h
	day := int(d / (h * time.Hour))
	millisecond := int(d/time.Millisecond) - (second * ms) - (minute * ms * s) - (hour * ms * s * m) - (day * ms * s * m * h)
	switch {
	case day > 0:
		return fmt.Sprintf("%dd%dh%dm%ds%dms", day, hour, minute, second, millisecond)
	case hour > 0:
		return fmt.Sprintf("%dh%dm%ds%dms", hour, minute, second, millisecond)
	case minute > 0:
		return fmt.Sprintf("%dm%ds%dms", minute, second, millisecond)
	case second > 0:
		return fmt.Sprintf("%ds%dms", second, millisecond)
	case millisecond > 0:
		return fmt.Sprintf("%dms", millisecond)
	default:
		return fmt.Sprint("0ms")
	}
}
