#!/bin/sh -e

e2e=$(basename $0)

usage() {
	echo "Usage: $e2e [flags]"
	echo "Flags:"
	echo "  -h, --help"
	echo "  --all"
	echo "  -t, --train"
	echo "  -s, --search"
	echo "  -c, --clean"
}

function build() {
	make build
}

function train() {
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

	echo "skip-gram with ns"
	./wego word2vec -i text8 -o example/word2vec_sg_ns.txt --model skip-gram --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20
	echo "skip-gram with hs"
	./wego word2vec -i text8 -o example/word2vec_sg_hs.txt --model skip-gram --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20
	echo "cbow with ns"
	./wego word2vec -i text8 -o example/word2vec_cbow_ns.txt --model cbow --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20
	echo "cbow with hs"
	./wego word2vec -i text8 -o example/word2vec_cbow_hs.txt --model cbow --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20

	echo "glove with sgd"
	./wego glove -d 50 -i text8 -o example/glove_sgd.txt --iter 10 --thread 12 --initlr 0.05 --min-count 5 -w 15 --solver sgd --verbose
	echo "glove with adagrad"
	./wego glove -d 50 -i text8 -o example/glove_adagrad.txt --iter 10 --thread 12 --initlr 0.05 --min-count 5 -w 15 --solver adagrad --verbose
}

function search() {
	echo "similarity search for skip-gram with ns"
	./wego search -i example/word2vec_sg_ns.txt microsoft
	echo "similarity search for skip-gram with hs"
	./wego search -i example/word2vec_sg_hs.txt microsoft
	echo "similarity search for cbow with ns"
	./wego search -i example/word2vec_cbow_ns.txt microsoft
	echo "similarity search for cbow with hs"
	./wego search -i example/word2vec_cbow_hs.txt microsoft

	echo "similarity search for glove with sgd"
	./wego search -i example/glove_sgd.txt microsoft
	echo "similarity search for glove with adagrad"
	./wego search -i example/glove_adagrad.txt microsoft
}

function clean() {
	rm example/word2vec_sg_ns.txt\
	   example/word2vec_sg_hs.txt\
	   example/word2vec_cbow_ns.txt\
	   example/word2vec_cbow_hs.txt\
	   example/glove_sgd.txt\
  	   example/glove_adagrad.txt\
	   wego
}

for OPT in "$@"; do
	case "$OPT" in
	'-h' | '--help')
		usage
		exit 1
		;;
	'--all')
		build
		train
		search
		;;
	'-t' | '--train')
		build
		train
		;;
	'-s' | '--search')
		build
		search
		;;
	'-c' | '--clean')
		clean
		;;
	-*)
		echo "$e2e: illegal option $1"
		usage
		exit 1
		;;
	esac
done
