#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : format-header.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/09/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import os
import os.path as osp
import glob

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger

logger = get_logger(__file__)
parser = JacArgumentParser()
parser.add_argument('--dir', default=os.getcwd())
parser.add_argument('--project', default=osp.basename(os.getcwd()))
parser.add_argument('-n', '--dry', action='store_true')
args = parser.parse_args()


HEADER = r"""#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : {file}
# Author : {author}
# Email  : {email}
# Date   : {date}
#
# This file is part of {project}.
# Distributed under terms of the MIT license.
"""

def log(*args, **kwargs):
    logger.info(*args, **kwargs)


def process(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    filetype = 'unk'
    fields = dict(project=args.project)
    
    for i, line in enumerate(lines):
        if i == 0:
            if line.startswith('#! /usr/bin/env'):
                filetype = 'vim'
            elif line.startswith('# -*- coding'):
                filetype = 'charm'
            else:
                filetype = 'unk'
    
        if i == 1 and filetype != 'unk':
            if not line.startswith('# '):
                filetype = 'unk'
    
        if filetype == 'unk':
            break
    
        line_trim = line[2:]
        if line_trim.startswith('File'):
            fields['file'] = line_trim[line_trim.find(':')+1:].strip()
            if fields['file'] != osp.basename(filename):
                log('  mismatched filename: full={}, actual={}, doc={}'.format(filename, osp.basename(filename), fields['file']))
                fields['file'] = osp.basename(filename)
        elif line_trim.startswith('Author'):
            fields['author'] = line_trim[line_trim.find(':')+1:].strip()
            if fields['author'] != 'Jiayuan Mao':
                log('  author assertion: full={}, author={}'.format(filename, fields['author']))
        elif line_trim.startswith('Email'):
            fields['email'] = line_trim[line_trim.find(':')+1:].strip()
            if fields['email'] != 'maojiayuan@gmail.com':
                log('  email assertion: full={}, email={}'.format(filename, fields['email']))
        elif line_trim.startswith('Date'):
            fields['date'] = line_trim[line_trim.find(':')+1:].strip()
            date = fields['date'].split('/')
            assert len(date) == 3
            if filetype == 'charm':
                date = (date[1], date[0], date[2])
            if int(date[0]) > 12 or (int(date[2]) == 2018 and int(date[0]) > 5):
                date = (date[1], date[0], date[2])
            date = list(date)
            date[0] = '{:02d}'.format(int(date[0]))
            date[1] = '{:02d}'.format(int(date[1]))
            if date[2] == '17':
                date[2] = '2017'
            if date[2] == '18':
                date[2] = '2018'
            fields['date'] = '/'.join(date)
    
        if i == 8 and filetype == 'vim':
            if not line_trim.startswith('Distributed'):
                log('  vim-typed file error: {}'.format(filename))
            break
        if i == 6 and filetype == 'charm':
            if not line_trim.startswith('This file'):
                log('  charm-typed file error: {}'.format(filename))
            break
    
        if not line.startswith('#'):
            break
    
    if filetype == 'unk':
        logger.warn('Unkown filetype.')
        return
    
    if filetype == 'vim':
        extras = lines[9:]
    elif filetype == 'charm':
        extras = lines[7:]
    
    assert len(fields) == 5

    if not args.dry:
        with open(filename, 'w') as f:
            f.write(HEADER.format(**fields))
            f.writelines(extras)


def main():
    logger.critical('Working directory: {}'.format(args.dir))
    logger.critical('Project name: {}'.format(args.project))

    for f in glob.glob('{}/**/*.py'.format(args.dir), recursive=True):
        logger.info('Process: "{}"'.format(f))
        process(f)


if __name__ == '__main__':
    main()

