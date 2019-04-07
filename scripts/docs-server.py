#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : docs-server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/21/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
Start the Jacinle docs server, which can be viewed through the browsers.
"""

import http.server
import socketserver
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', default=8080, type=int, help='Serving port.')
parser.add_argument('--make', action='store_true', help='Rebuild the HTMLs by make.')
parser.add_argument('--apidoc', action='store_true', help='Rebuild the RSTs by sphinx.')

args = parser.parse_args()


if __name__ == '__main__':
    dirname = os.path.dirname(sys.argv[0])

    docpath = os.path.realpath(os.path.join(dirname, '../', 'docs'))
    os.chdir(docpath)
    print('Chaning directory to {}'.format(docpath))

    if args.apidoc:
        os.system('rm -rf source/jac*.rst source/modules.rst source/vendors.rst && sphinx-apidoc -f -o source/ ../ -d 2 --module-first')
        os.system('rm -rf source/.vim*.rst')

    if args.make:
        os.system('make html')

    docpath = os.path.realpath(os.path.join(dirname, '../', 'docs', 'build', 'html'))
    os.chdir(docpath)
    print('Chaning directory to {}'.format(docpath))

    Handler = http.server.SimpleHTTPRequestHandler

    print('Serving at port {}.'.format(args.port))
    httpd = socketserver.TCPServer(("", args.port), Handler)
    httpd.serve_forever()

