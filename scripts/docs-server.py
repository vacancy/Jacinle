# -*- coding:utf8 -*-
# File   : docs-server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/25/17
# 
# This file is part of TensorArtist.

import http.server
import socketserver
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', dest='port', default=8080, type=int, help='Serving port.')
parser.add_argument('-m', '--make', dest='make', default=True, action='store_true', help='Run the makefile.')
port = parser.parse_args().port

if __name__ == '__main__':
    dirname = os.path.dirname(sys.argv[0])

    docpath = os.path.realpath(os.path.join(dirname, '../', 'docs'))
    os.chdir(docpath)
    print('Chaning directory to {}'.format(docpath))
    os.system('make html')

    docpath = os.path.realpath(os.path.join(dirname, '../', 'docs', '_build', 'html'))
    os.chdir(docpath)
    print('Chaning directory to {}'.format(docpath))

    Handler = http.server.SimpleHTTPRequestHandler
    
    print('Serving at port {}.'.format(port))
    httpd = socketserver.TCPServer(("", port), Handler)
    httpd.serve_forever()
