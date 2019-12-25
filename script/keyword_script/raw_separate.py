from os.path import expanduser
from os import makedirs
import re
from tqdm import tqdm
import argparse


def parse():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-s', '--source', dest='source',
      default = 'cnndm-pj/',
      help = 'source data')
  parser.add_argument('-d', '--dest', dest='dest',
      default = 'raw_summary/',
      help = 'destination directory prefix')
  parser.add_argument('-f', '--full', dest='full',
      action = 'store_true',
      help = 'process full text')
  return parser.parse_args()


def annotate(doc, summ):
  dlist = doc.split()
  sset = set(summ.split())
#return ' '.join([iw for w in dlist for iw in (w in sset)*['#'] + [w] + (w in sset)*['##']])
  return ' '.join([f'# {w} ##' if w in sset and not w in sw else w for w in dlist[:400]])


if __name__ == '__main__':
  home = expanduser("~") + '/'
  args = parse()
  dest_suf = ''
  source_dir = home + 'Data/' + args.source
  dest_dir = source_dir + args.dest + dest_suf
  makedirs(dest_dir, exist_ok=True)
  delimiter = '<summ-content>'
  modes = ['test', 'val', 'train']
  regex = r'<(/?)s>( ?)'
  
  for mode in modes:
    source = source_dir + mode + '.txt'
    dest_sum = dest_dir + mode + '.sum'
    
    with open(source) as so, open(dest_sum, 'w') as ds:
      lines = so.readlines()
      for i, l in enumerate(tqdm(lines)):
        if not args.full and i > 100:
          break
        s, d = l.split(delimiter)
        ssent = s.split(' </s> <s> ')
        ssent[0] = ssent[0][4:]
        ssent[-1] = ssent[-1][:-5]
        for j,sent in enumerate(ssent):
          ds.write(f'{i}-{j}\t{sent}\n')
