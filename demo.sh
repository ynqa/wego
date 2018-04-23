#!/bin/sh -e

if [ ! -e text8 ]; then
	echo "Download text8 corpus"
	if hash wget 2>/dev/null; then
		wget -q --show-progress http://mattmahoney.net/dc/text8.zip
	else
		curl --progress-bar -O http://mattmahoney.net/dc/text8.zip
	fi

	echo "Unzip text8.zip"
	unzip text8.zip
	rm text8.zip
fi

go run main.go word2vec --inputFile text8 --model skip-gram --optimizer ns --outputFile example/word_vectors_sg.txt -d 100 && \
    echo "Save trained vectors to example/word_vectors_sg.txt"
