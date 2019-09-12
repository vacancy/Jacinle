#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : render.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/22/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

"""
This file is adapted from the spaCy project: https://github.com/explosion/spaCy/blob/master/spacy/displacy/render.py.
The spaCy project is under MIT lisence:


The MIT License (MIT)

Copyright (C) 2016-2019 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import six
import uuid
from .templates import TPL_DEP_SVG, TPL_DEP_WORDS, TPL_DEP_ARCS, TPL_ENTS
from .templates import TPL_ENT, TPL_ENT_RTL, TPL_FIGURE, TPL_TITLE, TPL_PAGE
from .utils import minify_html, escape_html

DEFAULT_LANG = "en"
DEFAULT_DIR = "ltr"


class DependencyVisualizer(object):
    """Render dependency parses as SVGs."""

    style = "dep"

    def __init__(self, options={}):
        """Initialise dependency renderer.
        options (dict): Visualiser-specific options (compact, word_spacing,
            arrow_spacing, arrow_width, arrow_stroke, distance, offset_x,
            color, bg, font)
        """
        self.compact = options.get("compact", True)
        self.word_spacing = options.get("word_spacing", 45)
        self.arrow_spacing = options.get("arrow_spacing", 12 if self.compact else 20)
        self.arrow_width = options.get("arrow_width", 6 if self.compact else 10)
        self.arrow_stroke = options.get("arrow_stroke", 2)
        self.distance = options.get("distance", 150 if self.compact else 175)
        self.offset_x = options.get("offset_x", 50)
        self.color = options.get("color", "#000000")
        self.bg = options.get("bg", "#ffffff")
        self.font = options.get("font", "Arial")
        self.direction = options.get("direction", DEFAULT_DIR)
        self.lang = options.get("direction", DEFAULT_LANG)

    def render(self, parsed, page=False, minify=False):
        """Render complete markup.
        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (unicode): Rendered SVG or HTML markup.
        """
        # Create a random ID prefix to make sure parses don't receive the
        # same ID, even if they're identical
        id_prefix = uuid.uuid4().hex
        rendered = []
        for i, p in enumerate(parsed):
            render_id = "{}-{}".format(id_prefix, i)
            svg = self.render_svg(render_id, p["words"], p["arcs"])
            rendered.append(svg)

        # NB(Jiayuan Mao @ 08/22): slightly different from the original behavior in spaCy: return raw list of svgs.
        if page:
            content = "".join([TPL_FIGURE.format(content=svg) for svg in rendered])
            markup = TPL_PAGE.format(
                content=content, lang=self.lang, dir=self.direction
            )
            if minify:
                return minify_html(markup)
            return markup

        return rendered

    def render_simple_svg(self, text, arcs):
        if isinstance(text, six.string_types):
            text = text.split()
        output_words = [{'text': w, 'tag': ''} for w in text]

        output_arcs = list()
        for arc in arcs:
            if len(arc) == 3:
                start, end, label = arc
            else:
                assert len(arc) == 2
                start, end = arc
                label = ''
            arc = {'start': start, 'end': end, 'label': label, 'dir': 'left'}
            output_arcs.append(arc)

        id_prefix = uuid.uuid4().hex
        return self.render_svg(id_prefix + '-0', output_words, output_arcs)

    def render_svg(self, render_id, words, arcs):
        """Render SVG.
        render_id (int): Unique ID, typically index of document.
        words (list): Individual words and their tags.
        arcs (list): Individual arcs and their start, end, direction and label.
        RETURNS (unicode): Rendered SVG markup.
        """
        self.levels = self.get_levels(arcs)
        self.highest_level = len(self.levels)
        self.offset_y = self.distance / 2 * self.highest_level + self.arrow_stroke
        self.width = self.offset_x + len(words) * self.distance
        self.height = self.offset_y + 3 * self.word_spacing
        self.id = render_id
        words = [self.render_word(w['text'], w['tag'], i) for i, w in enumerate(words)]
        arcs = [
            self.render_arrow(a["label"], a["start"], a["end"], a["dir"], i)
            for i, a in enumerate(arcs)
        ]
        content = "".join(words) + "".join(arcs)
        return TPL_DEP_SVG.format(
            id=self.id,
            width=self.width,
            height=self.height,
            color=self.color,
            bg=self.bg,
            font=self.font,
            content=content,
            dir=self.direction,
            lang=self.lang,
        )

    def render_word(self, text, tag, i):
        """Render individual word.
        text (unicode): Word text.
        tag (unicode): Part-of-speech tag.
        i (int): Unique ID, typically word index.
        RETURNS (unicode): Rendered SVG markup.
        """
        y = self.offset_y + self.word_spacing
        x = self.offset_x + i * self.distance
        if self.direction == "rtl":
            x = self.width - x
        html_text = escape_html(text)
        return TPL_DEP_WORDS.format(text=html_text, tag=tag, x=x, y=y)

    def render_arrow(self, label, start, end, direction, i):
        """Render individual arrow.
        label (unicode): Dependency label.
        start (int): Index of start word.
        end (int): Index of end word.
        direction (unicode): Arrow direction, 'left' or 'right'.
        i (int): Unique ID, typically arrow index.
        RETURNS (unicode): Rendered SVG markup.
        """
        if start < 0 or end < 0:
            raise ValueError(f'Invalid arrow: start={start}, end={end}, label={label}.')
        level = self.levels.index(end - start) + 1
        x_start = self.offset_x + start * self.distance + self.arrow_spacing
        if self.direction == "rtl":
            x_start = self.width - x_start
        y = self.offset_y
        x_end = (
            self.offset_x
            + (end - start) * self.distance
            + start * self.distance
            - self.arrow_spacing * (self.highest_level - level) / 4
        )
        if self.direction == "rtl":
            x_end = self.width - x_end
        y_curve = self.offset_y - level * self.distance / 2
        if self.compact:
            y_curve = self.offset_y - level * self.distance / 6
        if y_curve == 0 and len(self.levels) > 5:
            y_curve = -self.distance
        arrowhead = self.get_arrowhead(direction, x_start, y, x_end)
        arc = self.get_arc(x_start, y, y_curve, x_end)
        label_side = "right" if self.direction == "rtl" else "left"
        return TPL_DEP_ARCS.format(
            id=self.id,
            i=i,
            stroke=self.arrow_stroke,
            head=arrowhead,
            label=label,
            label_side=label_side,
            arc=arc,
        )

    def get_arc(self, x_start, y, y_curve, x_end):
        """Render individual arc.
        x_start (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        y_curve (int): Y-corrdinate of Cubic Bézier y_curve point.
        x_end (int): X-coordinate of arrow end point.
        RETURNS (unicode): Definition of the arc path ('d' attribute).
        """
        template = "M{x},{y} C{x},{c} {e},{c} {e},{y}"
        if self.compact:
            template = "M{x},{y} {x},{c} {e},{c} {e},{y}"
        return template.format(x=x_start, y=y, c=y_curve, e=x_end)

    def get_arrowhead(self, direction, x, y, end):
        """Render individual arrow head.
        direction (unicode): Arrow direction, 'left' or 'right'.
        x (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        end (int): X-coordinate of arrow end point.
        RETURNS (unicode): Definition of the arrow head path ('d' attribute).
        """
        if direction == "left":
            pos1, pos2, pos3 = (x, x - self.arrow_width + 2, x + self.arrow_width - 2)
        else:
            pos1, pos2, pos3 = (
                end,
                end + self.arrow_width - 2,
                end - self.arrow_width + 2,
            )
        arrowhead = (
            pos1,
            y + 2,
            pos2,
            y - self.arrow_width,
            pos3,
            y - self.arrow_width,
        )
        return "M{},{} L{},{} {},{}".format(*arrowhead)

    def get_levels(self, arcs):
        """Calculate available arc height "levels".
        Used to calculate arrow heights dynamically and without wasting space.
        args (list): Individual arcs and their start, end, direction and label.
        RETURNS (list): Arc levels sorted from lowest to highest.
        """
        levels = set(map(lambda arc: arc["end"] - arc["start"], arcs))
        return sorted(list(levels))


defualt_renderer = DependencyVisualizer()


def visualize_list(parsed, options={}, page=False, minify=False):
    if len(options) == 0:
        renderer = defualt_renderer
    else:
        renderer = DependencyVisualizer(options)
    return renderer.render(parsed, page=page, minify=minify)


def visualize_simple_svg(text, arcs, options={}):
    if len(options) == 0:
        renderer = defualt_renderer
    else:
        renderer = DependencyVisualizer(options)
    return renderer.render_simple_svg(text, arcs)

