import csv


def write_to_csv(obj, fname):
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(obj)


def read_csv(fname):
    out = []
    with open(fname) as f:
        reader = csv.reader(f)
        for line in reader:
            out.append(line)
    return [item for sublist in out for item in sublist]
