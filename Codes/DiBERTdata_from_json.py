'''
Converts the structured dump (i.e, train_period_data.json) into
text file containing each post and comment per line, cleaned.
'''

import json
import cleaner
import sys
from multiprocessing import Process

def process_and_write_line(tree_obj, outfile):
    with open(outfile, 'a+') as _outfile:
        text = tree_obj['title']+'\n'+tree_obj['selftext']
        _outfile.write('{}\n'.format(cleaner.clean_pipeline(text, True)))
        for comment in tree_obj['comments']:
            if 'body' in comment.keys() and len(comment['body'].split())>5:
                _outfile.write('{}\n'.format(cleaner.clean_pipeline(comment['body'], False)))


def json_to_linedata(linelist, outfilename, process_id):
    print('Starting process {}'.format(process_id))
    for line in linelist:
        process_and_write_line(json.loads(line), outfilename)

if __name__=='__main__':
    infilename, outfilename, num_process = sys.argv[1], sys.argv[2], int(sys.argv[3])
    process_dict = {}
    with open(infilename) as _infile:
        lines = _infile.readlines()
    lines_per_process = len(lines)//(num_process-1)
    p_count = 1
    for line_index in range(0, len(lines), lines_per_process):
        process_lines = lines[line_index:line_index+lines_per_process]
        process_dict[line_index] = Process(target=json_to_linedata, args=(process_lines, outfilename, p_count,))
        process_dict[line_index].start()
        p_count +=1
    for key in process_dict.keys():
        process_dict[key].join()
