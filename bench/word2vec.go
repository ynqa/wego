package main

import (
	"os"

	"github.com/ynqa/word-embedding/builder"
)

func main() {
	w2v := builder.NewWord2VecBuilder()

	w2v.SetDimension(100).
		SetWindow(5).
		SetModel("cbow").
		SetNegativeSampleSize(5).
		SetBatchSize(10000)

	mod, _ := w2v.Build()
	file, _ := os.Open("text8")
	f, _ := mod.Preprocess(file)
	mod.Train(f)
}
