package main

import (
	"os"

	"github.com/ynqa/wego/pkg/model/modelutil/save"
	"github.com/ynqa/wego/pkg/model/word2vec"
)

func main() {
	model, err := word2vec.New(
		word2vec.WithWindow(5),
		word2vec.WithModel(word2vec.Cbow),
		word2vec.WithOptimizer(word2vec.NegativeSampling),
		word2vec.WithNegativeSampleSize(5),
		word2vec.Verbose(),
	)
	if err != nil {
		// failed to create word2vec.
	}

	input, _ := os.Open("text8")
	if err = model.Train(input); err != nil {
		// failed to train.
	}

	// write word vector.
	model.Save(os.Stdin, save.AggregatedVector)
}
