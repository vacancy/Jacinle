#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : format-header.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/09/2018
# 
# Distributed under terms of the MIT license.

import os.path as osp
import sys

def log(*args, **kwargs):
    print('[formatter]', *args, file=sys.stderr, **kwargs)

filename = sys.argv[1]
with open(filename) as f:
    lines = f.readlines()

filetype = 'unk'
fields = dict()

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
        log('  unk:', filename)
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
    sys.exit(0)

if filetype == 'vim':
    extras = lines[9:]
elif filetype == 'charm':
    extras = lines[7:]


assert len(fields) == 4

HEADER = r"""#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : {file}
# Author : {author}
# Email  : {email}
# Date   : {date}
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.
""".format(**fields)


with open(filename, 'w') as f:
    f.write(HEADER)
    f.writelines(extras)

