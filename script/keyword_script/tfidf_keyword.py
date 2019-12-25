from os.path import expanduser
from os import makedirs
import re
from tqdm import tqdm
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np
from scipy.sparse import save_npz, load_npz


def parse():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-s', '--source', dest='source',
      default = 'cnndm-pj/',
      help = 'source data')
  parser.add_argument('-d', '--dest', dest='dest',
      default = 'keyword/tfidf/from_doc/',
      help = 'destination directory prefix')
  parser.add_argument('-f', '--full', dest='full',
      action = 'store_true',
      help = 'process full text')
  parser.add_argument('-k', '--topk', dest='topK',
      default = 10,
      help = 'top k tfidf terms')
  parser.add_argument('--from-source', dest='from_source',
      action = 'store_true',
      help = 'tfidf from source texts')
  return parser.parse_args()

class LemmaTokenizer(object):
  def __init__(self):
    self.sp = spacy.load('en')
  def __call__(self, doc):
    spdoc = self.sp(doc)
    #spdoc = [token.lemma_ for token in spdoc if (len(token.lemma_) > 1) or (token.lemma_.isalnum())]
    spdoc = [token.lemma_ for token in spdoc if re.search(r'[a-z]', token.lemma_)]
    return spdoc
  def stopwords(self):
    sw = spacy.lang.en.stop_words.STOP_WORDS
    sw = list(set(token.lemma_ for token in self.sp(' '.join(sw))))
    sw += ["-pron-", "s",  "/s"]
    return sw 

def get_texts(source_dir, modes):
  delimiter = '<summ-content>'
  docs = []
  sums = []
  sizes = {}
  for mode in modes:
    source = source_dir + mode + '.txt'
    
    with open(source) as so:
      lines = so.readlines()
      sizes[mode] = len(lines)
      for i, l in enumerate(tqdm(lines)):
        if not args.full and i >= 10:
          sizes[mode] = 10
          break
        s, d = l.split(delimiter)
        docs.append(d)
        sums.append(s)
  return docs, sums, sizes

def fitted_vectorizer(texts, dest_dir):
  #vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer())
  tokenizer = LemmaTokenizer()
  vectorizer = TfidfVectorizer(
      stop_words=tokenizer.stopwords(), 
      tokenizer=tokenizer, 
      dtype=np.float32,
      max_features=80000,
      )
  print('fit')
  vectorizer.fit(texts)
  print('transform')
  vecs = vectorizer.transform(texts)
  save_npz(dest_dir + 'tfidf_sparse.npz', vecs)
  with open(dest_dir + 'tfidf_features.txt', 'w') as f:
    f.write('\n'.join(vectorizer.get_feature_names()))
  return vectorizer.get_feature_names(), vecs

def topk_terms(features, vecs, dest_dir, modes, sizes):
  vecs = vecs.toarray()  
  unsorted_topk = np.argpartition(-vecs, args.topK)[:, :args.topK]
  it = iter(zip(vecs, unsorted_topk))
  for mode in modes:
    dest = dest_dir + mode + '.txt'  
    with open(dest, 'w') as de:
      for _ in tqdm(range(sizes[mode])):
        vec, topk = it.__next__()
        values = vec[topk]
        sorted_topk = np.argsort(-values)
        out = [features[i] for i in topk[sorted_topk]]
        de.write(' '.join(out) + '\n')


def topk_terms_with_csr(features, vecs, dest_dir, modes, sizes):
  sum_size = 0  
  for mode in modes:
    dest = dest_dir + mode + '.txt'
    with open(dest, 'w') as de:
      for r in tqdm(range(sum_size, sum_size+sizes[mode])):
        vec = vecs.getrow(r).toarray().squeeze()
        unsorted_topk = np.argpartition(-vec, args.topK)[:args.topK]
        values = vec[unsorted_topk]
        sorted_topk = np.argsort(-values)
        out = [features[i] for i in unsorted_topk[sorted_topk]]
        de.write(' '.join(out) + '\n')
    sum_size += sizes[mode]
  

def load_data(dest_dir):
  vecs = load_npz(dest_dir + 'tfidf_sparse.npz')
  print(type(vecs))
  with open(dest_dir + 'tfidf_features.txt') as f:
    features = f.read().splitlines()
  return features, vecs

if __name__ == '__main__':
  home = expanduser("~") + '/'
  args = parse()

  dest_suf = ''
  source_dir = home + 'Data/' + args.source
  dest_dir = source_dir + args.dest + dest_suf
  makedirs(dest_dir, exist_ok=True)
  modes = ['test', 'val', 'train']
  
  docs, sums, sizes = get_texts(source_dir, modes)
  print(sizes)
  if args.from_source:
    features, vecs = fitted_vectorizer(docs, dest_dir)
  else:
    features, vecs = load_data(dest_dir)
  # topk_terms(features, vecs, dest_dir, modes, sizes)
  topk_terms_with_csr(features, vecs, dest_dir, modes, sizes)
