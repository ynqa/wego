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

make build

echo "skip-gram with ns"
./wego word2vec -i text8 -o example/word2vec_sg_ns.txt --model skip-gram --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20
echo "skip-gram with hs"
./wego word2vec -i text8 -o example/word2vec_sg_hs.txt --model skip-gram --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20
echo "cbow with ns"
./wego word2vec -i text8 -o example/word2vec_cbow_ns.txt --model cbow --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20
echo "cbow with hs"
./wego word2vec -i text8 -o example/word2vec_cbow_hs.txt --model cbow --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20

echo "skip-gram with ns"
./wego distance -i example/word2vec_sg_ns.txt microsoft
echo "skip-gram with hs"
./wego distance -i example/word2vec_sg_hs.txt microsoft
echo "cbow with ns"
./wego distance -i example/word2vec_cbow_ns.txt microsoft
echo "cbow with hs"
./wego distance -i example/word2vec_cbow_hs.txt microsoft

rm example/word2vec_sg_ns.txt\
  example/word2vec_sg_hs.txt\
  example/word2vec_cbow_ns.txt\
  example/word2vec_cbow_hs.txt

echo "glove with sgd"
./wego glove -d 50 -i text8 -o example/glove_sgd.txt --iter 10 --thread 12 --initlr 0.05 --min-count 5 -w 15 --solver sgd --verbose
echo "glove with adagrad"
./wego glove -d 50 -i text8 -o example/glove_adagrad.txt --iter 10 --thread 12 --initlr 0.05 --min-count 5 -w 15 --solver adagrad --verbose

echo "glove with sgd"
./wego distance -i example/glove_sgd.txt microsoft
echo "glove with adagrad"
./wego distance -i example/glove_adagrad.txt microsoft

rm example/glove_sgd.txt\
  example/glove_adagrad.txt

rm wego
