import argparse
import csv



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="the path to the result file")
    parser.add_argument('--save_path', type=str, default='results.csv')
    parser.add_argument('--name', type=str, default='baseline', help='name of the input row')

    args = parser.parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as f:
        with open(args.save_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            data = [args.name]
            for line in f.readlines():
                if line[0].isalpha():
                    line = line.split('=')
                    if len(line)==2:
                        data.append(float(line[1])*100)
            writer.writerow(data)

