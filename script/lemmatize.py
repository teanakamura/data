from os.path import expanduser
from os import makedirs
import re
from tqdm import tqdm
import argparse
import constant
import contextlib
import spacy
import time
import easydict

def parse():
    args = easydict.EasyDict(dict(
        data = 'cnndm-pj',
        dest = 'keyword/tfidf/from_doc_ngram',
        full = True,
        topK = 10,
        from_source = False
        ))
    return args

class Lemmatizer():
    def __init__(self):
        self.sp = spacy.load('en')
    def __call__(self, doc):
        return ' '.join([token.lemma_ for token in self.sp(doc)])

if __name__ == '__main__':
    args = parse()

    data_dir = f'{expanduser("~")}/Data/{args.data}'
    source_dir = data_dir
    dest_dir = f'{data_dir}/lemmatize'
    makedirs(dest_dir, exist_ok=True)
    print(f'create data in {dest_dir}')

    delimiter = '<summ-content>'
    regex = r'<(/?)s>( ?)'
    repl = r'<\1t>\2'

    modes = ['test', 'val', 'train']
    for mode in modes:
        source = f'{source_dir}/{mode}.txt'
        dest_sum = f'{dest_dir}/{mode}.sum'
        dest_doc = f'{dest_dir}/{mode}.doc'

        with open(source) as so:
            n_lines = len(so.readlines())

        lemmatizer = Lemmatizer()

        with open(source) as so, open(dest_sum, 'w') as ds, open(dest_doc, 'w') as dd:
            for i, l in enumerate(tqdm(so, total=n_lines)):
                s, d = l.split(delimiter)
                s = re.sub(regex, repl, s)
                d = lemmatizer(d)
                ds.write(s + '\n')
                dd.write(d.rstrip() + '\n')
