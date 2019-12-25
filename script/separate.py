from os.path import expanduser
from os import makedirs
import re
from tqdm import tqdm
import argparse
import constant
import contextlib
import spacy
import time

class Default():
    data = 'cnndm-pj'
    dest_suf = ''
    tfidf = 'keyword/tfidf/from_sum'
    stopword = 'keyword/rake/stopwords.txt'


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data', dest='data',
        default = Default.data,
        help = 'source data.') 
    parser.add_argument('-f', '--full', dest='full',
        action = 'store_true',
        help = 'process full text.')
    parser.add_argument('-a', '--annotation', dest='annt', 
        default = None,
        help = '''
            annotate source keywords.
            select tfidf or stopword.
        ''')
    parser.add_argument('--dest-suf', dest='dest_suf',
        default = Default.dest_suf,
        help = 'destination directory suffix.')
    return parser.parse_args()


class StopwordAnnt():
    def __init__(self):
        with open(f'{data_dir}/{Default.stopword}') as f:
            self.sw = set(f.read().split())
    def __call__(self, doc, summ):
        dlist = doc.split()
        sset = set(summ.split())
        #return ' '.join([iw for w in dlist for iw in (w in sset)*['#'] + [w] + (w in sset)*['##']])
        return ' '.join([f'## {w} ###' if w in sset and not w in self.sw else w for w in dlist[:400]])


class TfidfAnnt():
    def __init__(self):
        self.sp = spacy.load('en')
    def __call__(self, doc, keys):
        keys = set(keys.split())
        return ' '.join([f'## {token.text} ###' if token.lemma_ in keys else token.text for token in self.sp(doc)[:400]])
    

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
    
    global data_dir
    data_dir = f'{expanduser("~")}/Data/{args.data}'
    source_dir = data_dir
    size = 'full' if args.full else 'small'
    dest_f = f'{args.annt}_annt' if args.annt else 'base'
    dest_f += f'_{args.dest_suf}' if args.dest_suf else ''
    dest_dir = f'{data_dir}/{size}/{dest_f}'
    makedirs(dest_dir, exist_ok=True)
    print(f'create data in {dest_dir}')
    modes = ['test', 'val', 'train']
    
    delimiter = '<summ-content>'
    regex = r'<(/?)s>( ?)'
    repl = r'<\1t>\2'
  
    for mode in modes:
        source = f'{source_dir}/{mode}.txt'
        dest_sum = f'{dest_dir}/{mode}.sum'
        dest_doc = f'{dest_dir}/{mode}.doc'
        stopwords = source_dir + 'rake/stopwords.txt'
    
        if args.annt == 'stopword':
            annotator = StopwordAnnt()
        elif args.annt == 'tfidf':
            annotator = TfidfAnnt()
        
        with open(source) as so:
            n_lines = len(so.readlines())

        with open(source) as so, open(dest_sum, 'w') as ds, open(dest_doc, 'w') as dd:
            with open(f'{data_dir}/{Default.tfidf}/{mode}.txt') if args.annt=='tfidf' else dummy_context_mgr() as ke: 
                for i, (l, k) in enumerate(tqdm(zip(so, ke), total=n_lines)):
                    if not args.full and i > 100: break
                    s, d = l.split(delimiter)
                    s = re.sub(regex, repl, s)
                    if args.annt == 'stopword':
                        d = annotator(d, s)
                    elif args.annt == 'tfidf':
                        d = annotator(d, k)
                    ds.write(s + '\n')
                    dd.write(d.rstrip() + '\n')
