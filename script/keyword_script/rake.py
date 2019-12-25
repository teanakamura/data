from rake_nltk import Rake, Metric
from os.path import expanduser
from tqdm import tqdm
import re

home = expanduser('~')
delimiter = '<summ-content>'
stopwords_path = home + '/Data/cnndm-pj/rake/stopwords.txt'
source_dir = home + '/Data/cnndm-pj/'
dest_dir = home + '/Data/cnndm-pj/rake/doc/'
modes = ['test'] #, 'val', 'train']

with open(stopwords_path) as f:
  stopwords = f.read().split()

sep = ' <ph> '
r = Rake(
        stopwords = stopwords, 
        punctuations = ["``", "''", "--", "`", "'", ",", ".", "?"],
    ) #ranking_metric=Metric.WORD_DEGREE)

for mode in modes:
  ph_list = []
  source = source_dir + mode + '.txt'
  dest = dest_dir + mode + '.txt'
  with open(source) as so:
    lines = so.readlines()
    for i,l in enumerate(tqdm(lines)):
      # if i > 10:
      #  break
      s, d = l.split(delimiter)
      d = re.sub(r"``|''", "", d)
      r.extract_keywords_from_text(d)
      ph = r.get_ranked_phrases()
      ph_list.append(sep.join(ph))
  with open(dest, 'w') as de:
    de.write('\n'.join(ph_list))
