#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : index.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/23/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.
from jacweb.web import route, JacRequestHandler

@route(r'/')
class IndexHandler(JacRequestHandler):
    def get(self):
        nr_visited = 0
        if self.session is not None:
            nr_visited = self.session.get('nr_visited', 0)
            self.session['nr_visited'] = nr_visited + 1
        self.render('index.html', nr_visited=nr_visited, headers=self.request.headers)

