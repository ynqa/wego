#!/bin/sh -e

e2e=$(basename $0)

usage() {
	echo "Usage: $e2e [flags]"
	echo "Flags:"
	echo "  -h, --help"
	echo "  -a, --all"
	echo "  --all-word2vec"
	echo "  --all-glove"
	echo "  --all-lexvec"
	echo "  -t, --train"
	echo "  --train-word2vec"
	echo "  --train-glove"
	echo "  --train-lexvec"
	echo "  -s, --search"
	echo "  --search-word2vec"
	echo "  --search-glove"
	echo "  --search-lexvec"
}

function build() {
	make build
}

function get_corpus() {
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
}

function train_word2vec() {
	echo "train: skip-gram with ns"
	./wego word2vec -i text8 -o example/word2vec_sg_ns.txt \
		--model skip-gram --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --save-vec add --thread 20
	echo "train: skip-gram with hs"
	./wego word2vec -i text8 -o example/word2vec_sg_hs.txt \
		--model skip-gram --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20
	echo "train: cbow with ns"
	./wego word2vec -i text8 -o example/word2vec_cbow_ns.txt \
		--model cbow --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --save-vec add --thread 20
	echo "train: cbow with hs"
	./wego word2vec -i text8 -o example/word2vec_cbow_hs.txt \
		--model cbow --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --thread 20
}

function train_glove() {
	echo "train: glove with sgd"
	./wego glove -d 50 -i text8 -o example/glove_sgd.txt \
		--iter 10 --thread 12 --initlr 0.05 --min-count 5 -w 15 --solver sgd --save-vec add --verbose
	echo "train: glove with adagrad"
	./wego glove -d 50 -i text8 -o example/glove_adagrad.txt \
		--iter 10 --thread 12 --initlr 0.05 --min-count 5 -w 15 --solver adagrad --save-vec add --verbose
}

function train_lexvec() {
	echo "train: lexvec"
	./wego lexvec -d 50 -i text8 -o example/lexvec.txt \
		--iter 3 --thread 12 --initlr 0.05 --min-count 5 -w 5 --rel ppmi --save-vec add --verbose
}

function search_word2vec() {
	echo "similarity search: skip-gram with ns"
	./wego search -i example/word2vec_sg_ns.txt microsoft
	echo "similarity search: skip-gram with hs"
	./wego search -i example/word2vec_sg_hs.txt microsoft
	echo "similarity search: cbow with ns"
	./wego search -i example/word2vec_cbow_ns.txt microsoft
	echo "similarity search: cbow with hs"
	./wego search -i example/word2vec_cbow_hs.txt microsoft
}

function search_glove() {
	echo "similarity search: glove with sgd"
	./wego search -i example/glove_sgd.txt microsoft
	echo "similarity search: glove with adagrad"
	./wego search -i example/glove_adagrad.txt microsoft
}

function search_lexvec() {
	echo "similarity search: lexvec"
	./wego search -i example/lexvec.txt microsoft
}

for OPT in "$@"; do
	case "$OPT" in
	'-h' | '--help')
		usage
		exit 1
		;;
	'-a' | '--all')
		build
		get_corpus
		train_word2vec
		train_glove
		train_lexvec
		search_word2vec
		search_glove
		search_lexvec
		;;
	'--all-word2vec')
		build
		get_corpus
		train_word2vec
		search_word2vec
		;;
	'--all-glove')
		build
		get_corpus
		train_glove
		search_glove
		;;
	'--all-lexvec')
		build
		get_corpus
		train_lexvec
		search_lexvec
		;;
	'-t' | '--train')
		build
		get_corpus
		train_word2vec
		train_glove
		train_lexvec
		;;
	'--train-word2vec')
		build
		get_corpus
		train_word2vec
		;;
	'--train-glove')
		build
		get_corpus
		train_glove
		;;
	'--train-lexvec')
		build
		get_corpus
		train_lexvec
		;;
	'-s' | '--search')
		build
		search_word2vec
		search_glove
		search_lexvec
		;;
	'--search-word2vec')
		build
		search_word2vec
		;;
	'--search-glove')
		build
		search_glove
		;;
	'--search-lexvec')
		build
		search_lexvec
		;;
	-*)
		echo "$e2e: illegal option $1"
		usage
		exit 1
		;;
	esac
done
