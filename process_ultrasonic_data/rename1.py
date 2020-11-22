import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="dir path")
    parser.add_argument("--xls", default='D:/DeskTop/Ultrasonic_v3/names.xls', help="excel filename",)
    args = parser.parse_args()

    path = args.path if args.path else os.getcwd()

    if args.xls:
        df = pd.read_excel(os.path.join(path, args.xls))
        df = df.drop(columns=[df.columns[0]])
        new_names = df['new'].tolist()
        # for i in range(len(new_names)):
        #     if new_names[i][0] == 'L':
        #         x = new_names[i].split('_')
        #         if len(x) == 5:
        #             if x[3][-1] in ['0','1','2','3','4','5','6','7','8','9']:
        #                 x[3] = x[3][:-1]
        #         else:
        #             (name,suffix) = x[3].split('.')
        #             if name[-1] in ['0','1','2','3','4','5','6','7','8','9']:
        #                 x[3] = name[:-1]+'_'+name[-1]+'.'+suffix
        #         new_names[i] = '_'.join(x)
        # print(new_names)
        # df['new'] = new_names
        # df.to_excel(os.path.join(os.path.split(args.xls)[0], 'new_names.xls'))

        # for i in range(len(new_names)):
        #     if new_names[i][0] == 'L':
        #         x = new_names[i].split('_')
        #         if len(x) == 4:
        #             (name, suffix) = x[3].split('.')
        #             x[3] = name + '_1' + '.' + suffix
        #         new_names[i] = '_'.join(x)
        # print(new_names)
        # df['new'] = new_names
        # df.to_excel(os.path.join(os.path.split(args.xls)[0], 'new_names.xls'))

        name_index = {}
        person_set = set()
        no_set=set()
        person_list = []
        no_list = []
        for i in range(len(new_names)):
            if new_names[i][0] == 'L':
                x = new_names[i].split('_')
                if x[2] not in no_list:
                    no_list.append(x[2])
                    person_list.append(x[3])
                person_set.add(x[3])
                no_set.add(x[2])
        print(len(person_set),len(no_set))
        import collections
        print([item for item, count in collections.Counter(person_list).items() if count > 1])

        index = 1
        with open(os.path.join(os.path.split(args.xls)[0], 'name_dict.txt'), 'w') as f:
            for no, person in zip(no_list, person_list):
                if index<10:
                    name_index[no+person]='n00'+str(index)
                    f.write(f'n00{str(index)}\t{no+person}\n')
                elif index<100:
                    name_index[no+person] = 'n0' + str(index)
                    f.write(f'n0{str(index)}\t{no+person}\n')
                else:
                    name_index[no+person] = 'n' + str(index)
                    f.write(f'n{str(index)}\t{no+person}\n')
                index = index+1
            f.close()
        print(name_index)
        for i in range(len(new_names)):
            if new_names[i][0] == 'L':
                x = new_names[i].split('_')
                new_names[i] = '_'.join([x[0],name_index[x[2]+x[3]],x[1],x[4]])
        print(new_names)
        df['new'] = new_names
        df.to_excel(os.path.join(os.path.split(args.xls)[0], 'new_names.xls'))


if __name__ == '__main__':
    main()