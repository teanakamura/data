#!/bin/bash


DATA=../cnndm-pj/
SUBWORD_DIR=${DATA}subword-nmt/
NUM_OPERATIONS=20000
case $1 in
  learn) 
    echo learn
    FILES=(train.txt val.txt 'test.txt')
    for i in $(seq 0 $((${#FILES[@]} - 1))); do
      FILES[$i]=${DATA}${FILES[$i]}
    done
    for f in "${FILES[@]}"; do
      echo $f
    done
    cat ${FILES[@]} | subword-nmt learn-bpe -s ${NUM_OPERATIONS} -o ${SUBWORD_DIR}codes${NUM_OPERATIONS}.txt
    ;;
  apply) 
    echo apply
    if [ ! -f ${DATA}$2 ]; then
      echo "No file ${DATA}/$2"
      exit 1
    fi
    CODES=${SUBWORD_DIR}/codes${NUM_OPERATIONS}.txt
    DROPOUT=0.1
    GLOOSSARIES="'<s>' '</s>' '<summ-content>'"
    # subword-nmt apply-bpe -c ${CODES} --dropout ${DROPOUT} < ${DATA}$2 > ${SUBWORD_DIR}$2.bpe
    subword-nmt apply-bpe -c ${CODES} < ${DATA}$2 > ${SUBWORD_DIR}$2.bpe
    ;;
  make-vocab)
    echo make vocab
    if [ ! -f ${SUBWORD_DIR}$2 ]; then
      echo "No file ${SUBWORD_DIR}$2"
      exit 1
    fi
    if [ -z $3 ]; then
      echo "specify vocabulary file to output"
      exit 1
    fi
    subword-nmt get-vocab < ${SUBWORD_DIR}$2 > ${SUBWORD_DIR}$3
    ;;
  count-vocab)
    echo count vocab
    if [ ! -f ${SUBWORD_DIR}$2 ]; then
      echo "No file ${SUBWORD_DIR}/$2"
      exit 1
    fi
    subword-nmt get-vocab < ${SUBWORD_DIR}$2 | wc -l
    ;;
  *) echo "input learn or apply";;
esac


