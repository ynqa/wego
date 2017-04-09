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

package fileio

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Save writes the text to output file.
func Save(outputPath string, s interface{}) error {
	dir := extractDir(outputPath)

	if err := os.MkdirAll("."+string(filepath.Separator)+dir, 0777); err != nil {
		return err
	}

	file, err := os.Create(outputPath)

	if err != nil {
		return err
	}
	w := bufio.NewWriter(file)

	defer func() {
		w.Flush()
		file.Close()
	}()

	w.WriteString(fmt.Sprintf("%v", s))

	return nil
}

// FileExists returns whether the file exists or not.
func FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func extractDir(path string) string {
	e := strings.Split(path, "/")
	return strings.Join(e[:len(e)-1], "/")
}
