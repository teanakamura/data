from os.path import expanduser
home = expanduser('~')

dict_path = home + '/fairseq/workplace/data-bin/cnndm/dict.doc.txt'
stop_words = []
with open(dict_path) as f:
  for l in f:
    w, c = l.split()
    if int(c) < 100000: break
    stop_words.append(w)

stopwords_path = home + '/Data/cnndm-pj/rake/stopwords.txt'
with open(stopwords_path, 'w') as f:
  f.write(' '.join(stop_words))
