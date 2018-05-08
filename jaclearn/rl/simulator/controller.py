#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/23/2017
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import tkinter as tk
import threading
import time

from PIL import Image, ImageTk


class Controller(object):
    def __init__(self, title='Controller', fps=100):
        self.records = dict()
        self.last_key = None

        self.tk_root = None
        self.tk_canv = None
        self.tk_image = None

        self.__lock = threading.Lock()
        self.__title = title

        self.__current_lock = threading.Lock()
        self.__current = None
        self.__fps = fps
        self.__stop_event = threading.Event()

    def update_title(self, title):
        assert self.tk_root is None
        self.__title = title

    def update(self, img):
        with self.__current_lock:
            self.__current = img

    def get_last_key(self):
        while True:
            self.__lock.acquire()
            res = self.last_key
            if res is None:
                self.__lock.release()
                time.sleep(0.1)
            else:
                self.last_key = None
                self.__lock.release()
                return res

    def test_key(self, key):
        return self.records.get(key, False)

    def mainloop(self):
        while True:
            if self.__current is None:
                time.sleep(1)
                if self.__stop_event.is_set():
                    return
            else:
                break

        self.__create_tk()
        self.__update_image_thread()
        self.tk_root.mainloop()

    def quit(self):
        self.__stop_event.set()
        if self.tk_root is not None:
            self.tk_root.quit()

    def __create_tk(self):
        self.tk_root = tk.Tk()
        self.tk_root.title(self.__title)

        self.__update_image()
        self.tk_canv = tk.Label(self.tk_root, image=self.tk_image, bd=0)
        self.tk_canv.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
        self.tk_canv.focus_set()
        self.tk_canv.bind("<KeyPress>", self.__press)
        self.tk_canv.bind("<KeyRelease>", self.__release)

    def __update_image(self):
        with self.__current_lock:
            img = Image.fromarray(self.__current)
        imgtk = ImageTk.PhotoImage(img)
        self.tk_image = imgtk

    def __update_image_thread(self):
        self.__update_image()
        self.tk_canv.config(image=self.tk_image)
        self.tk_root.update_idletasks()
        self.tk_root.after(int(1000 / self.__fps), self.__update_image_thread)

    def __press(self, event):
        with self.__lock:
            self.records[event.keysym_num] = True
            self.last_key = event.keysym_num

    def __release(self, event):
        with self.__lock:
            self.records[event.keysym_num] = False
