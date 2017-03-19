package model

import "fmt"

var cnt int

const batch = 1000

func printTrace(denominator int) {
	cnt++
	if cnt%batch == 0 {
		fmt.Printf("Train Progress: %d / %d\n", cnt, denominator)
	}
}
