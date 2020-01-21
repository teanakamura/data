from os.path import expanduser
from os import makedirs
import re
from tqdm import tqdm
import argparse
import constant
import contextlib
import spacy
import time
from collections import Counter

class Default():
    data = 'cnndm-pj'
    dest_suf = ''
    keyword_pattern = 'tfidf'
    keypath = 'keyword/tfidf/from_doc_ngram'
    stoppath = 'keyword/stopwords.txt'

def parse():
    parent_parser = argparse.ArgumentParser(description='', add_help=False)
    parent_parser.add_argument('-d', '--data', dest='data',
        default = Default.data,
        help = 'source data.') 
    parent_parser.add_argument('-f', '--full', dest='full',
        action = 'store_true',
        help = 'process full text.')
    parent_parser.add_argument('--dest-suf', dest='dest_suf',
        default = Default.dest_suf,
        help = 'destination directory suffix.')
    parent_parser.add_argument('--truncate', dest='trun',
        default = 400,
        help = 'truncate words.')
    
    parser = argparse.ArgumentParser(description='', parents=[parent_parser])
    subparsers = parser.add_subparsers(description='annotation pattern', dest='annt')
    
    sw_parser = subparsers.add_parser('stopword', parents=[parent_parser])
    sw_parser.add_argument('--stopword', dest='stoppath',
        default = Default.stoppath,
        help = 'stop word file path.')
    sw_parser.add_argument('-l', '--lemmatize', dest='lemma',
        action = 'store_true',
        help = 'lemmatize when comparing a document word with a summry word')
    
    key_parser = subparsers.add_parser('keyword', parents=[parent_parser])
    key_parser.add_argument('-l', '--lemmatize', dest='lemma',
        action = 'store_true',
        help = 'lemmatize when comparing a document word with a summry word')
    key_parser.add_argument('-k', '--keyword-pattern', dest='key', 
        default = Default.keyword_pattern,
        help = '''
            annotate source keywords.
            select from {tfidf}.
        ''')
    key_parser.add_argument('--keypath', dest='keypath', 
        default = Default.keypath,
        help = 'source keyword directory.')
    return parser.parse_args()


class StopwordAnnt():
    def __init__(self, stoppath):
        with open(stoppath) as f:
            if args.lemma:
                self.sp = spacy.load('en')
            if args.lemma:
                self.sw = {token.lemma_ for token in self.sp(f.read())} ######
            else:
                self.sw = set(f.read().split())
    def __call__(self, doc, summ):
        if args.lemma:
            sset = {token.lemma_ for token in self.sp(summ)[:args.trun]}
            return ' '.join([f'## {t.text} ###' 
                             if t.lemma_ in sset and not t.lemma_ in self.sw else t.text 
                             for t in self.sp(doc)[:args.trun]]) #######
        else:
            sset = set(summ.split())
            #return ' '.join([iw for w in dlist for iw in (w in sset)*['#'] + [w] + (w in sset)*['##']])
            return ' '.join([f'## {w} ###' 
                             if w in sset and not w in self.sw else w 
                             for w in doc.split()[:args.trun]])


class TfidfAnnt():
    def __init__(self):
        self.sp = spacy.load('en')
    def __call__(self, doc, keys):
        keys = set(keys.split())
        return ' '.join([f'## {token.text} ###' \
                         if token.lemma_ in keys else token.text \
                         for token in self.sp(doc)[:args.trun]])
    

@contextlib.contextmanager
def dummy_context_mgr():
    #yield DummyGenerator()
    yield dummy_generator()
    
class DummyGenerator():
    def __iter__(self):
        return self
    def __next__(self):
        return None
def dummy_generator():
    while True:
        yield None
        

if __name__ == '__main__':
    args = parse()
    
    data_dir = f'{expanduser("~")}/Data/{args.data}'
    source_dir = data_dir
    size = 'full' if args.full else 'small'
    if args.annt == 'stopword':
        dest_f = 'stopword_annt'
    elif args.annt == 'keyword':
        dest_f = f'{args.key}_annt'
    else:
        dest_f = 'base'
    dest_f += f'_{args.dest_suf}' if args.dest_suf else ''
    dest_dir = f'{data_dir}/{size}/{dest_f}'
    makedirs(dest_dir, exist_ok=True)
    print(f'create data in {dest_dir}')
    
    with open(f'{dest_dir}/args.txt', 'w') as f:
        f.write(str(args))
    
    delimiter = '<summ-content>'
    regex = r'<(/?)s>( ?)'
    repl = r'<\1t>\2'
    
    if args.annt == 'stopword':
        annotator = StopwordAnnt(f'{data_dir}/{args.stoppath}')
    elif args.annt == 'keyword':
        if args.key == 'tfidf':
            annotator = TfidfAnnt()

    modes = ['test','val', 'train']
    for mode in modes:
        source = f'{source_dir}/{mode}.txt'
        dest_sum = f'{dest_dir}/{mode}.sum'
        dest_doc = f'{dest_dir}/{mode}.doc'
        
        with open(source) as so:
            n_lines = len(so.readlines())

        with open(source) as so, \
                open(dest_sum, 'w') as ds, \
                open(dest_doc, 'w') as dd, \
                open(f'{data_dir}/{args.keypath}') if getattr(args, 'keypath', None) else dummy_context_mgr() as ke \
                open(f'{data_dir}/{args.lemmapath}') if getattr(args, 'lemmapath', None) else dummy_context_mgr() as le:
            for i, (l, k, ls) in enumerate(tqdm(zip(so, ke, le), total=n_lines)):
                if not args.full and i > 100: break
                s, d = l.split(delimiter)
                s = re.sub(regex, repl, s)
                if args.annt == 'stopword':
                    d = annotator(d, s)
                elif args.annt == 'keyword':
                    d = annotator(d, k, ls)
                else:
                    d = ' '.join(d.split()[:args.trun])
                ds.write(s + '\n')
                dd.write(d.rstrip() + '\n')
