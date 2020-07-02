#!/bin/bash

CURRENT_DIR=`pwd`
DATA=~/Data/cnndm-pj/full/tfidf_annt_modified
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

if [[ $1 =~ (apply|apply-all|apply-annt|apply-annt-all) ]]; then
  CODES=${SUBWORD_DIR}/codes${NUM_OPERATIONS}.txt
  #CODES=~/Data/cnndm-pj/full/tfidf_annt/subword-nmt/codes${NUM_OPERATIONS}.txt
  DROPOUT=0.1
  #GLOOSSARIES="'<s>' '</s>' '<summ-content>'"
  GLOOSSARIES="'<t>' '</t>'"
fi

if [ $1 = 'apply-annt' ] || [ $1 = 'apply-annt-all' ]; then
  EXE_PY=~/Data/script/custom_subword_nmt/subword_nmt.py
  ADD_FILES=(`ls *.add`)
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
    if echo $2 | grep -qE '\.(doc|sum)$'; then
      # subword-nmt apply-bpe -c ${CODES} --dropout ${DROPOUT} < ${DATA}/$2 > ${SUBWORD_DIR}/$2.bpe
      echo "subword-nmt apply-bpe -c ${CODES} < ${DATA}/$2 > ${SUBWORD_DIR}/$2.bpe"
      subword-nmt apply-bpe -c ${CODES} < ${DATA}/$2 > ${SUBWORD_DIR}/$2.bpe
    else
      echo "subword-nmt apply-bpe -c ${CODES} < ${DATA}/$2.doc > ${SUBWORD_DIR}/$2.bpe"
      subword-nmt apply-bpe -c ${CODES} < ${DATA}/$2.doc > ${SUBWORD_DIR}/$2.bpe
      echo "subword-nmt apply-bpe -c ${CODES} < ${DATA}/$2.sum > ${SUBWORD_DIR}/$2.bpe"
      subword-nmt apply-bpe -c ${CODES} < ${DATA}/$2.sum > ${SUBWORD_DIR}/$2.bpe
    fi
    ;;

  apply-all)
    echo apply all
    for f in "${FILES[@]}"; do
      echo $f
      subword-nmt apply-bpe -c ${CODES} < $f > ${SUBWORD_DIR}/$f.bpe
    done
    ;;

  apply-annt)
    echo apply with annotation
    BASENAME=${2%%.*}
    if [ ! -f ${DATA}/${BASENAME}.add ]; then
      echo "No file ${DATA}/$2.add"
      exit 1
    fi
    if echo $2 | grep -qE '\.doc$' || test $2 = $BASENAME; then
      echo "python $EXE_PY apply-bpe -c $CODES -i $DATA/$BASENAME.doc -o $SUBWORD_DIR/$BASENAME.doc.bpe -a $DATA/$BASENAME.add -A $SUBWORD_DIR/$BASENAME.add.bpe"
      python $EXE_PY apply-bpe -c $CODES -i $DATA/$BASENAME.doc -o $SUBWORD_DIR/$BASENAME.doc.bpe -a $DATA/$BASENAME.add -A $SUBWORD_DIR/$BASENAME.add.bpe
    fi
    if echo $2 | grep -qE '\.sum' || test $2 = $BASENAME; then
      echo "subword-nmt apply-bpe -c ${CODES} -i ${DATA}/$BASENAME.sum -o ${SUBWORD_DIR}/$BASENAME.sum.bpe"
      subword-nmt apply-bpe -c ${CODES} -i ${DATA}/$2 -o ${SUBWORD_DIR}/$2.bpe
    fi
    ;;

  apply-annt-all)
    echo apply all with annotation
    for f in "${ADD_FILES[@]}"; do
      BASENAME=${f%%.*}
      echo $BASENAME
      echo "python $EXE_PY apply-bpe -c $CODES -i $DATA/$BASENAME.doc -o $SUBWORD_DIR/$BASENAME.doc.bpe -a $DATA/$BASENAME.add -A $SUBWORD_DIR/$BASENAME.add.bpe"
      python $EXE_PY apply-bpe -c $CODES -i $DATA/$BASENAME.doc -o $SUBWORD_DIR/$BASENAME.doc.bpe -a $DATA/$BASENAME.add -A $SUBWORD_DIR/$BASENAME.add.bpe
      echo "subword-nmt apply-bpe -c ${CODES} -i ${DATA}/$BASENAME.sum -o ${SUBWORD_DIR}/$BASENAME.sum.bpe"
      subword-nmt apply-bpe -c ${CODES} -i ${DATA}/$BASENAME.sum -o ${SUBWORD_DIR}/$BASENAME.sum.bpe
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


