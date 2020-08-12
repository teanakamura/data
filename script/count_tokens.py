import argparse
import os

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('type')
    parser.add_argument('--data', '-d',
            default = 'cnndm-pj')
    parser.add_argument('--subword', '-s',
            action = 'store_true')
    parser.add_argument('--data-path', '-p')
    return parser.parse_args()

def get_data_path(args):
    script_relpath = os.path.dirname(__file__)
    data_abspath = os.path.abspath(os.path.join(
        script_relpath,
        f'../{args.data}/full/{args.type}',
        'subword-nmt' if args.subword else ''))
    return data_abspath

if __name__ == '__main__':
    args = parse()
    print(args)
    data_path = args.data_path or get_data_path(args)
    print(data_path)
    
    modes = ['train', 'val', 'test']
    exts = ['doc', 'sum']
    for ext in exts:
        ext_acc = 0
        for mode in modes:
            mode_acc = 0
            mode_max = 0
            count = 0
            file_name = f'{mode}.{ext}'
            file_path = os.path.join(data_path, file_name)
            with open(file_path) as f:
                for l in f:
                    ntokens = len(l.split())
                    #if ntokens > 1000 and ext == 'sum':
                    #from IPython import embed; embed()
                    mode_max = max(mode_max, ntokens)
                    mode_acc += ntokens
                    count += 1
            print(f'{file_name}:\tmax: {mode_max}\tave: {mode_acc/count}\tsents: {count}')





