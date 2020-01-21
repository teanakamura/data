from os.path import expanduser
from os import makedirs
import re
from tqdm import tqdm
import argparse
import constant
import contextlib
import spacy
import time
from collections import Counter, defaultdict
from time import time
from multiprocessing import Process, Manager

class Default():
    data = 'cnndm-pj'
    dest_suf = ''
    keyword_pattern = 'tfidf'
    keypath = 'keyword/tfidf/from_doc_ngram'
    lemmapath = 'lemmatize'
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

# from easydict import EasyDict
# def parse():
#     args = EasyDict(
#         data = Default.data,
#         full = True,
#         dest_suf = '',
#         trun = 400,
#         lemma = True,
#         key = Default.keyword_pattern,
#         keypath = Default.keypath,
#         annt = 'keyword'
#     )
#     return args


class BaseProcess():
    def __init__(self, delimiter, regex, repl, trun):
        self.delimiter = delimiter
        self.regex = regex
        self.repl = repl
        self.trun = trun
    def __call__(self, sumdoc, _=None):
        summ, doc = sumdoc.split(delimiter)
        summ = re.sub(regex, repl, summ)
        doc = ' '.join(doc.split()[:self.trun])
        return summ, doc

class StopwordAnnt():
    def __init__(self, preprocess, stoppath, lemma):
        self.preprocess = preprocess
        self.lemma = lemma
        with open(stoppath) as f:
            if lemma:
                self.sp = spacy.load('en')
                self.sw = {token.lemma_ for token in self.sp(f.read())} ######
            else:
                self.sw = set(f.read().split())
    def __call__(self, sumdoc, _=None):
        summ, doc = self.preprocess(sumdoc)
        if self.lemma:
            sset = {token.lemma_ for token in self.sp(summ)}
            return summ, ' '.join([f'## {t.text} ###' 
                             if t.lemma_ in sset and not t.lemma_ in self.sw else t.text 
                             for t in self.sp(doc)]) #######
        else:
            sset = set(summ.split())
            #return ' '.join([iw for w in dlist for iw in (w in sset)*['#'] + [w] + (w in sset)*['##']])
            return summ, ' '.join([f'## {w} ###' 
                             if w in sset and not w in self.sw else w 
                             for w in doc.split()[:args.trun]])

class TfidfNgramAnnt():
    def __init__(self, preprocess):
        self.sp = spacy.load('en')
        self.preprocess = preprocess
    def __call__(self, sumdoc, keys):
        summ, doc = self.preprocess(sumdoc)

        sp_doc = self.sp(doc.rstrip())
        first_word_dict = defaultdict(list)
        keys = keys.rstrip()
        if keys:
            for vk in keys.split(', '):
                v, *ks = vk.split()
                if float(v) < 0.05:
                    break
                first_word_dict[ks[0]].append(ks)

        it = iter(sp_doc)  #sp_doc.__iter__()
        tmp = []
        res = []
        while True:
            try:
                token = next(it) if not tmp else tmp.pop() #it.__next__()
            except StopIteration:
                break
            if token.lemma_ in first_word_dict:
                for key in first_word_dict[token.lemma_]:
                    try:
                        for _ in range(2-len(tmp)):
                            tmp.append(next(it))
                    except StopIteration:
                        pass
                    tmp = tmp[::-1]  # [nntoken, ntoken]
                    if len(tmp) >= len(key)-1:
                        for i in range(1, len(key)):
                            if not tmp[-i].lemma_ == key[i]:
                                break
                        else:
                            out_list = [token.text]
                            for _ in range(len(key)-1):
                                out_list.append(tmp.pop().text)
                            res.append(f'## {" ".join(out_list)} ###')
                            break
            else:
                res.append(token.text)
        return summ, ' '.join(res)

class ProcessFunc():
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, so, ke, start, end, returned_sum, returned_doc, pnum):
        for sumdoc, keys, idx in zip(so[start:end], ke[start:end], range(start, end)):
            if pnum == 0:
                print(f'process0: {idx+1:>{self.digits}}/{self.size_pp}', end='\r', flush=True)  # flushしないと表示されないことがある．
            summ, doc = self.processor(sumdoc, keys)
            returned_sum[idx] = summ
            returned_doc[idx] = doc
    def set_size_pp(self, size_pp):
        self.size_pp = size_pp
        self.digits = len(str(size_pp))


@contextlib.contextmanager
def dummy_context_mgr(n):
    # yield DummyGenerator()
    # yield dummy_generator()
    yield dummy_limited_generator(n)

class DummyGenerator():
    def __iter__(self):
        return self
    def __next__(self):
        return None
def dummy_generator():
    while True:
        yield None
def dummy_limited_generator(n):
    for _ in range(n):
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

    processor = BaseProcess(delimiter, regex, repl, args.trun)
    if args.annt == 'stopword':
        processor = StopwordAnnt(processor, f'{data_dir}/{args.stoppath}', args.lemma)
    elif args.annt == 'keyword':
        if args.key == 'tfidf':
            processor = TfidfNgramAnnt(processor)
    processfunc = ProcessFunc(processor)

    modes = ['test','val', 'train']
    totaltime = 0
    for mode in modes:
        size = None if args.full else 100
        source_txt = f'{source_dir}/{mode}.txt'
        dest_sum = f'{dest_dir}/{mode}.sum'
        dest_doc = f'{dest_dir}/{mode}.doc'
        # key_txt = f'{data_dir}/{args.keypath}/{mode}.txt' if getattr(args, 'keypath', None) else None
        key_txt = f'{data_dir}/{args.keypath}/{mode}_sumfilter.txt' if getattr(args, 'keypath', None) else None
        
        with open(source_txt) as so:
            n_lines = len(so.readlines())
        
        size = min(size, n_lines) if size else n_lines
        print(f'{mode}: total size: {size}')
        num_process = 32
        process_list = []
        size_pp = (size-1) // num_process + 1  ## size per process
        processfunc.set_size_pp(size_pp)
        with open(source_txt) as so, \
                open(dest_sum, 'w') as ds, \
                open(dest_doc, 'w') as dd, \
                open(key_txt) if key_txt else dummy_context_mgr(n_lines) as ke, \
                Manager() as m:
            so_list = m.list(list(so))
            ke_list = m.list(list(ke))
            returned_sum = m.list([None]*size)
            returned_doc = m.list([None]*size)
            t1 = time()
            for i in range(num_process):
                start = size_pp*i
                end = min(size_pp*(i+1), size)
                p = Process(target=processfunc, args=(so_list, ke_list, start, end, returned_sum, returned_doc, i))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            t2 = time()
            print()
            ds.write('\n'.join(returned_sum).rstrip() + '\n')  ## wc -l で数えやすくするため
            dd.write('\n'.join(returned_doc).rstrip() + '\n')
        totaltime += t2-t1
    print(totaltime)


    
    
    
    
    
    
    
    
    
    



class TfidfAnnt():
    def __init__(self):
        self.sp = spacy.load('en')
    def __call__(self, doc, keys):
        keys = set(keys.split())
        return ' '.join([f'## {token.text} ###' \
                         if token.lemma_ in keys else token.text \
                         for token in self.sp(doc)[:args.trun]])
