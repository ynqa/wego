#!/bin/bash -e

E2E=$(basename $0)
function usage() {
	cat <<EOF
Usage:
	${E2E} [flags]

Flags:
	-a, --all         run all jobs for all models
	--all-word2vec    run all jobs for word2vec
	--all-glove       run all jobs for glove
	--all-lexvec      run all jobs for lexvec
	-t, --train       run train job for all models
	--train-word2vec  run train job for word2vec
	--train-glove     run train job for glove
	--train-lexvec    run train job for lexvec
	-s, --search      run search job for all models
	--search-word2vec run search job for word2vec
	--search-glove    run search job for glove
	--search-lexvec   run search job for lexvec
	-h, --help        print usage
EOF
}

function build() {
	go build
}

function clean_examples() {
	rm -rf *.txt
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
		echo "train: skipgram with ns"
	./wego word2vec -i text8 -o word2vec_sg_ns.txt \
		--model skipgram --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --save-type agg --goroutines 20 --batch 10000
	echo "train: skipgram with hs"
	./wego word2vec -i text8 -o word2vec_sg_hs.txt \
		--model skipgram --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --goroutines 20 --batch 10000
	echo "train: cbow with ns"
	./wego word2vec -i text8 -o word2vec_cbow_ns.txt \
		--model cbow --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --save-type agg --goroutines 20 --batch 10000
	echo "train: cbow with hs"
	./wego word2vec -i text8 -o word2vec_cbow_hs.txt \
		--model cbow --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --goroutines 20 --batch 10000

	# echo "train: skipgram with ns"
	# ./wego word2vec -i text8 -o word2vec_sg_ns.txt --in-memory \
	# 	--model skipgram --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --save-type agg --goroutines 20 --batch 100000
	# echo "train: skipgram with hs"
	# ./wego word2vec -i text8 -o word2vec_sg_hs.txt --in-memory \
	# 	--model skipgram --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --goroutines 20 --batch 100000
	# echo "train: cbow with ns"
	# ./wego word2vec -i text8 -o word2vec_cbow_ns.txt --in-memory \
	# 	--model cbow --optimizer ns -d 100 -w 5 --verbose --iter 3 --min-count 5 --save-type agg --goroutines 20 --batch 100000
	# echo "train: cbow with hs"
	# ./wego word2vec -i text8 -o word2vec_cbow_hs.txt --in-memory \
	# 	--model cbow --optimizer hs -d 100 -w 5 --verbose --iter 3 --min-count 5 --goroutines 20 --batch 100000
}

function train_glove() {
	echo "train: glove with sgd"
	./wego glove -d 50 -i text8 -o glove_sgd.txt --in-memory \
		--iter 5 --goroutines 12 --initlr 0.01 --min-count 5 -w 10 --solver sgd --save-type agg --verbose
	echo "train: glove with adagrad"
	./wego glove -d 50 -i text8 -o glove_adagrad.txt --in-memory \
		--iter 5 --goroutines 12 --initlr 0.05 --min-count 5 -w 10 --solver adagrad --save-type agg --verbose
}

function train_lexvec() {
	echo "train: lexvec"
	./wego lexvec -d 50 -i text8 -o lexvec.txt \
		--iter 3 --goroutines 12 --initlr 0.05 --min-count 5 -w 5 --rel ppmi --save-type agg --verbose
	# echo "train: lexvec"
	# ./wego lexvec -d 50 -i text8 -o lexvec.txt --in-memory \
	# 	--iter 3 --goroutines 12 --initlr 0.05 --min-count 5 -w 5 --rel ppmi --save-type agg --verbose
}

function search_word2vec() {
	echo "similarity search: skipgram with ns"
	./wego query -i word2vec_sg_ns.txt microsoft
	echo "similarity search: skipgram with hs"
	./wego query -i word2vec_sg_hs.txt microsoft
	echo "similarity search: cbow with ns"
	./wego query -i word2vec_cbow_ns.txt microsoft
	echo "similarity search: cbow with hs"
	./wego query -i word2vec_cbow_hs.txt microsoft
}

function search_glove() {
	echo "similarity search: glove with sgd"
	./wego query -i glove_sgd.txt microsoft
	echo "similarity search: glove with adagrad"
	./wego query -i glove_adagrad.txt microsoft
}

function search_lexvec() {
	echo "similarity search: lexvec"
	./wego query -i lexvec.txt microsoft
}

for OPT in "$@"; do
	case "${OPT}" in
	'-h' | '--help')
		usage
		exit 1
		;;
	'-a' | '--all')
		clean_examples
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
		clean_examples
		build
		get_corpus
		train_word2vec
		search_word2vec
		;;
	'--all-glove')
		clean_examples
		build
		get_corpus
		train_glove
		search_glove
		;;
	'--all-lexvec')
		clean_examples
		build
		get_corpus
		train_lexvec
		search_lexvec
		;;
	'-t' | '--train')
		clean_examples
		build
		get_corpus
		train_word2vec
		train_glove
		train_lexvec
		;;
	'--train-word2vec')
		clean_examples
		build
		get_corpus
		train_word2vec
		;;
	'--train-glove')
		clean_examples
		build
		get_corpus
		train_glove
		;;
	'--train-lexvec')
		clean_examples
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
		echo "${E2E}: illegal option $1"
		usage
		exit 1
		;;
	esac
done
