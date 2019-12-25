from os.path import expanduser
import re

home = expanduser("~") + '/'
source_dir = home + 'Data/cnndm-pj/'
dest_dir = home + 'Data/cnndm-pj/cnndm_dot/'
delimiter = '<summ-content>'
modes = ['test', 'val', 'train']
regex = r'. </s>'

def repl(matchobj):
  if matchobj.group(0) == '. </s>':
    repl.i += 1
    return f'.{repl.i} '
  else:
    return ''

for mode in modes:
  source = source_dir + mode + '.txt'
  dest_sum = dest_dir + mode + '.sum'
  dest_doc = dest_dir + mode + '.doc'

  with open(source) as so, open(dest_sum, 'w') as ds, open(dest_doc, 'w') as dd:
    for i, l in enumerate(so):
      if i > 100:
        break
      s, d = l.split(delimiter)
      repl.i = 0
      s = re.sub(regex, repl, s)
      ds.write(s + '\n')
      dd.write(d)
