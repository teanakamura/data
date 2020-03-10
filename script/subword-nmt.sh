#!/bin/bash

CURRENT_DIR=`pwd`
DATA=~/Data/cnndm-pj/full/tfidf_annt
cd $DATA
SUBWORD_DIR=${DATA}/subword-nmt/
mkdir -p $SUBWORD_DIR
NUM_OPERATIONS=20000

if [ $1 = learn ] || [ $1 = 'apply-all' ]; then
  #FILES=(train.doc val.doc 'test.doc' train.sum val.sum 'test.sum')
  FILES=(`ls *.doc *.sum`)
  ##for i in $(seq 0 $((${#FILES[@]} - 1))); do  ## for bash
  #for i in $(seq 1 ${#FILES[@]}); do  ## for zsh
  #  FILES[$i]=${DATA}/${FILES[$i]}
  #done
fi

if [ $1 = apply ] || [ $1 = 'apply-all' ]; then
  CODES=${SUBWORD_DIR}/codes${NUM_OPERATIONS}.txt
  #CODES=~/Data/cnndm-pj/full/tfidf_annt/subword-nmt/codes${NUM_OPERATIONS}.txt
  DROPOUT=0.1
  #GLOOSSARIES="'<s>' '</s>' '<summ-content>'"
  GLOOSSARIES="'<t>' '</t>'"
fi

case $1 in
  learn) 
    echo learn
    for f in "${FILES[@]}"; do
      echo $f
    done
    cat ${FILES[@]} | subword-nmt learn-bpe -s ${NUM_OPERATIONS} -o ${SUBWORD_DIR}/codes${NUM_OPERATIONS}.txt
    ;;
  apply) 
    echo apply
    if [ ! -f ${DATA}/$2 ]; then
      echo "No file ${DATA}/$2"
      exit 1
    fi
    # subword-nmt apply-bpe -c ${CODES} --dropout ${DROPOUT} < ${DATA}/$2 > ${SUBWORD_DIR}/$2.bpe
    subword-nmt apply-bpe -c ${CODES} < ${DATA}/$2 > ${SUBWORD_DIR}/$2.bpe
    ;;
  apply-all)
    echo apply all
    for f in "${FILES[@]}"; do
      echo $f
      subword-nmt apply-bpe -c ${CODES} < $f > ${SUBWORD_DIR}/$f.bpe
    done
    ;;
  make-vocab)
    echo make vocab
    if [ ! -f ${SUBWORD_DIR}/$2 ]; then
      echo "No file ${SUBWORD_DIR}/$2"
      exit 1
    fi
    if [ -z $3 ]; then
      echo "specify vocabulary file to output"
      exit 1
    fi
    subword-nmt get-vocab < ${SUBWORD_DIR}/$2 > ${SUBWORD_DIR}/$3
    ;;
  count-vocab)
    echo count vocab
    if [ ! -f ${SUBWORD_DIR}/$2 ]; then
      echo "No file ${SUBWORD_DIR}/$2"
      exit 1
    fi
    subword-nmt get-vocab < ${SUBWORD_DIR}/$2 | wc -l
    ;;
  rename-all)
    cd ${SUBWORD_DIR}
    rename s/.bpe//g ./*
    ;;
  *) echo "input learn, apply or apply-all";;
esac

cd $CURRENT_DIR


