with open('./cv-valid-train.csv', 'r') as f_in:
    with open('./cv-valid-train/train.tsv', 'w+') as f_out:
        data = 'path,sentence' + f_in.read()[13:]
        data = '\n'.join(['client_id,' + line for line in data.split('\n')])
        data = data.replace(',', '\t')
        f_out.write(data)

with open('./cv-valid-test.csv', 'r') as f_in:
    with open('./cv-valid-test/test.tsv', 'w+') as f_out:
        data = 'path,sentence' + f_in.read()[13:]
        data = '\n'.join(['client_id,' + line for line in data.split('\n')])
        data = data.replace(',', '\t')
        f_out.write(data)