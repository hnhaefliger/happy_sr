print('reformatting training labels...')

with open('./cv-valid-train.csv', 'r') as f_in:
    with open('./cv-valid-train/train.tsv', 'w+') as f_out:
        data = 'path,sentence' + f_in.read()[13:]
        data = '\n'.join(['client_id,' + line for line in data.split('\n') if (len(line.split(',')[0].split('/')[-1]) > 4 or 'path,sentence' in line)])
        data = data.replace(',', '\t')
        f_out.write(data)

print('done reformatting training labels.')
print('reformatting evaluation labels...')

with open('./cv-valid-test.csv', 'r') as f_in:
    with open('./cv-valid-test/test.tsv', 'w+') as f_out:
        data = 'path,sentence' + f_in.read()[13:]
        data = '\n'.join(['client_id,' + line for line in data.split('\n') if (len(line.split(',')[0].split('/')[-1]) > 4 or 'path,sentence' in line)])
        data = data.replace(',', '\t')
        f_out.write(data)

print('done reformatting evaluation labels.\n')
