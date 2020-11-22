import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="dir path")
    parser.add_argument("--xls", default=None, help="excel filename",)
    args = parser.parse_args()

    path = args.path if args.path else os.getcwd()

    if args.xls:
        df = pd.read_excel(os.path.join(path, args.xls))
        old_names = df['old'].values.tolist()
        new_names = df['new'].values.tolist()
        if len(new_names) != len(set(new_names)):
            print('Contains duplicate names!')
        else:
            for i in range(len(old_names)):
                if old_names[i] != new_names[i]:
                    os.rename(os.path.join(path, old_names[i]), os.path.join(path, new_names[i]))

    else:
        files = os.listdir(path)
        files_ = []
        for file in files:
            if '.' not in file:
                files_.append(file)
                continue
            [name, suffix] = file.split('.')
            blank_num = name.count(' ')
            append = '' if blank_num == 0 else '_'+str(blank_num)
            files_.append(name.replace(' ', '') + append + '.' + suffix)
        data = {'old':files, 'new':files_}
        df = pd.DataFrame(data)
        df.to_excel(os.path.join(path, 'names.xls'))


if __name__ == '__main__':
    x='aa_11_bb'
    c = x.split('_')
    print(c)