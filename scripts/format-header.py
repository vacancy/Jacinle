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
import time

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger

logger = get_logger(__file__)
parser = JacArgumentParser()
parser.add_argument('--dir', default=os.getcwd())
parser.add_argument('--include', nargs='*')
parser.add_argument('--exclude', nargs='*')
parser.add_argument('--project', default=osp.basename(os.getcwd()))
parser.add_argument('-n', '--dry', action='store_true')
parser.add_argument('-f', '--force', action='store_true')
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

    if len(lines) == 0:
        return

    filetype = 'unk'
    fields = dict()
    fields['project'] = args.project

    i = 0
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
                if not args.force:
                    filetype = 'unk'
                else:
                    filetype = 'force'

        if filetype == 'unk':
            break

        if not line.startswith('#'):
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
            if int(date[0]) > 12:
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
        if i == 6 and filetype == 'charm':
            if not line_trim.startswith('This file'):
                log('  charm-typed file error: {}'.format(filename))

    if lines[i].startswith('#'):
        i += 1

    if filetype == 'unk':
        logger.warning('Unkown filetype.')
        return

    extras = lines[i:]

    if len(fields) != 5:
        logger.warning('Incomplete header.')

        fields.setdefault('file', osp.basename(filename))
        fields.setdefault('author', 'Jiayuan Mao')
        fields.setdefault('email', 'maojiayuan@gmail.com')
        fields.setdefault('date', time.strftime('%m/%d/%Y', time.localtime(osp.getctime(filename))))

    if not args.dry:
        with open(filename, 'w') as f:
            f.write(HEADER.format(**fields))
            if len(extras):
                f.writelines(extras)


def myglob(root, exclude):
    output = list()
    exclude = set(exclude)
    def myglob_inner(dir):
        nonlocal output

        full_dir = osp.join(root, dir) if dir is not None else root
        for subdir in os.listdir(full_dir):
            if subdir.startswith('.'):
                continue
            subdir = osp.join(dir, subdir) if dir is not None else subdir
            if subdir in exclude:
                continue
            full_subdir = osp.join(root, subdir)
            if osp.isfile(full_subdir) and full_subdir.endswith('.py'):
                output.append(full_subdir)
            elif osp.isdir(full_subdir):
                myglob_inner(subdir)
    try:
        myglob_inner(None)
        return output
    finally:
        del myglob_inner


def main():
    logger.critical('Working directory: {}'.format(args.dir))
    logger.critical('Project name: {}'.format(args.project))

    files = list()

    if args.include:
        for f in args.files:
            if osp.isdir(f):
                files.extend(glob.glob('{}/**/*.py'.format(f), recursive=True))
            else:
                assert osp.isfile(f), f
                files.append(f)

    if args.exclude:
        files.extend(myglob(args.dir, args.exclude))
    else:
        files.extend(glob.glob('{}/**/*.py'.format(args.dir), recursive=True))

    for f in files:
        logger.info('Process: "{}"'.format(f))
        process(f)


if __name__ == '__main__':
    main()

