#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : session.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/23/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import hmac
import uuid
import hashlib

__all__ = ['Session', 'SessionManagerBase']


class Session(dict):
    def __init__(self, session_manager, request_handler):
        """
        Initialize the session.

        Args:
            self: (todo): write your description
            session_manager: (todo): write your description
            request_handler: (todo): write your description
        """
        self._session_manager = session_manager
        self._request_handler = request_handler

        session_id, session_data = self._session_manager.get(self._request_handler)
        self._identifier = session_id
        super().__init__(session_data)

    @property
    def identifier(self):
        """
        The identifier : the identifier

        Args:
            self: (todo): write your description
        """
        return self._identifier

    def save(self):
        """
        Save a new session_manager object.

        Args:
            self: (todo): write your description
        """
        self._session_manager.set(self._request_handler, self.identifier, dict(self))


class SessionManagerBase(object):
    def __init__(self, secret):
        """
        Initialize a secret.

        Args:
            self: (todo): write your description
            secret: (str): write your description
        """
        self._secret = secret
        super().__init__()

    @property
    def secret(self):
        """
        The secret asnodes.

        Args:
            self: (todo): write your description
        """
        return self._secret

    def new(self, request_handler):
        """
        Creates a new request.

        Args:
            self: (todo): write your description
            request_handler: (todo): write your description
        """
        return Session(self, request_handler)

    def get(self, request_handler):
        """
        Returns the request handler.

        Args:
            self: (todo): write your description
            request_handler: (todo): write your description
        """
        raise NotImplementedError()

    def set(self, request_handler, session_id, data):
        """
        Sets the request.

        Args:
            self: (todo): write your description
            request_handler: (todo): write your description
            session_id: (str): write your description
            data: (array): write your description
        """
        raise NotImplementedError()

    def _generate_id(self):
        """
        Generate a unique hash.

        Args:
            self: (todo): write your description
        """
        new_id = hashlib.sha256((self.secret + str(uuid.uuid4())).encode('utf-8'))
        return new_id.hexdigest()

    def _generate_hmac(self, session_id):
        """
        Generate hmac - sha256 hash.

        Args:
            self: (todo): write your description
            session_id: (str): write your description
        """
        if type(session_id) is not bytes:
            session_id = session_id.encode('utf-8')
        return hmac.new(session_id, self.secret.encode('utf-8'), hashlib.sha256).hexdigest()

