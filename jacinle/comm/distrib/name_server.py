#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : name_server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
import queue
import threading
import time

import zmq

from jacinle.comm.distrib import _configs
from jacinle.concurrency import zmq_utils as utils
from jacinle.logging import get_logger
from jacinle.utils.registry import CallbackRegistry

logger = get_logger(__file__)

__all__ = ['NameServer', 'run_name_server']


class NameServerControllerStorage(object):
    def __init__(self):
        self._all_peers = {}
        self._all_peers_req = {}
        self._outputs = collections.defaultdict(list)
        self._inputs = collections.defaultdict(list)

    def register(self, info, req_sock):
        identifier = info['uid']
        assert identifier not in self._all_peers
        self._all_peers[identifier] = {
            'uid': info['uid'],
            'ctl_protocal': info['ctl_protocal'],
            'ctl_addr': info['ctl_addr'],
            'ctl_port': info['ctl_port'],
            'meta': info.get('meta', {}),
            'outputs': [],
            'inputs': [],
            'last_heartbeat': time.time()
        }
        self._all_peers_req[identifier] = req_sock

    def register_outputs(self, info):
        controller = info['uid']
        assert controller in self._all_peers
        record = self._all_peers[controller]
        for i in record['outputs']:
            self._inputs[i].remove(controller)
        record['outputs'] = info['outputs']
        for i in record['outputs']:
            self._inputs[i].append(controller)

    def register_inputs(self, info):
        controller = info['uid']
        assert controller in self._all_peers
        record = self._all_peers[controller]
        for i in record['inputs']:
            self._outputs[i].remove(controller)
        record['inputs'] = info['inputs']
        for i in record['inputs']:
            self._outputs[i].append(controller)

    def unregister(self, identifier):
        if identifier in self._all_peers:
            info = self._all_peers.pop(identifier)
            for i in info['inputs']:
                self._outputs[i].remove(identifier)
            for i in info['outputs']:
                self._inputs[i].remove(identifier)
            return info, self._all_peers_req.pop(identifier)
        return None

    def get(self, identifier):
        return self._all_peers.get(identifier, None)

    def items(self):
        return self._all_peers.items()

    def contains(self, identifier):
        return identifier in self._all_peers

    def get_req_sock(self, identifier):
        return self._all_peers_req.get(identifier, None)

    def get_inputs(self, name):
        return self._outputs[name]

    def get_outputs(self, name):
        return self._inputs[name]


class NameServer(object):
    def __init__(self, host, port, protocal):
        self.storage = NameServerControllerStorage()
        self._addr = '{}://{}:{}'.format(protocal, host, port)
        self._context_lock = threading.Lock()
        self._context = zmq.Context()
        self._router = self._context.socket(zmq.ROUTER)
        self._poller = zmq.Poller()
        self._dispatcher = CallbackRegistry()
        self._req_socks = set()
        self._all_threads = list()
        self._control_send_queue = queue.Queue()

    def mainloop(self):
        self.initialize()
        try:
            self._all_threads.append(threading.Thread(target=self.main, name='name-server-main'))
            self._all_threads.append(threading.Thread(target=self.main_cleanup, name='name-server-cleanup'))
            for i in self._all_threads:
                i.start()
        finally:
            self.finalize()

    def initialize(self):
        self._router.bind(self._addr)
        self._poller.register(self._router, zmq.POLLIN)

        self._dispatcher.register(_configs.Actions.NS_REGISTER_CTL_REQ, self._on_ns_register_controller_req)
        self._dispatcher.register(_configs.Actions.NS_REGISTER_OUTPUTS_REQ, self._on_ns_register_outputs_req)
        self._dispatcher.register(_configs.Actions.NS_REGISTER_INPUTS_REQ, self._on_ns_register_inputs_req)

        self._dispatcher.register(_configs.Actions.NS_HEARTBEAT_REQ, self._on_ns_heartbeat_req)

        self._dispatcher.register(_configs.Actions.NS_NOTIFY_OPEN_REP, lambda msg: None)
        self._dispatcher.register(_configs.Actions.NS_NOTIFY_CLOSE_REP, lambda msg: None)

    def finalize(self):
        for i in self._all_threads:
            i.join()

        for sock in self._req_socks:
            utils.graceful_close(sock)
        utils.graceful_close(self._router)
        if not self._context.closed:
            self._context.destroy(0)

    def main_cleanup(self):
        while True:
            with self._context_lock:
                now = time.time()
                for k, v in list(self.storage.items()):
                    if (now - v['last_heartbeat']) > _configs.NS_CLEANUP_WAIT:
                        info, req_sock = self.storage.unregister(k)
                        self._poller.unregister(req_sock)
                        utils.graceful_close(req_sock)
                        self._req_socks.remove(req_sock)

                        # TODO(Jiayuan Mao @ 05/08): use controller's heartbeat.
                        all_peers_to_inform = set()
                        for i in info['inputs']:
                            all_peers_to_inform = all_peers_to_inform.union(self.storage.get_outputs(i))
                        for i in info['outputs']:
                            all_peers_to_inform = all_peers_to_inform.union(self.storage.get_inputs(i))
                        logger.debug('Inform died: {}.'.format(str(all_peers_to_inform)))

                        for peer in all_peers_to_inform:
                            self._control_send_queue.put({
                                'sock': self.storage.get_req_sock(peer),
                                'countdown': _configs.CTL_CTL_SND_COUNTDOWN,
                                'payload': {
                                    'action': _configs.Actions.NS_NOTIFY_CLOSE_REQ,
                                    'uid': k
                                },
                            })
                        logger.info('Unregister timeout controller {}.'.format(k))
            time.sleep(_configs.NS_CLEANUP_WAIT)

    def main(self):
        while True:
            with self._context_lock:
                socks = dict(self._poller.poll(50))
                self._main_do_send()
                self._main_do_recv(socks)

    def _main_do_send(self):
        nr_send = self._control_send_queue.qsize()
        for i in range(nr_send):
            job = self._control_send_queue.get()
            rc = utils.req_send_json(job['sock'], job['payload'], flag=zmq.NOBLOCK)
            if not rc:
                job['countdown'] -= 1
                if job['countdown'] >= 0:
                    self._control_send_queue.put(job)
                else:
                    logger.warning('Drop job: {}.'.format(str(job)))

    def _main_do_recv(self, socks):
        if self._router in socks and socks[self._router] == zmq.POLLIN:
            for identifier, msg in utils.iter_recv(utils.router_recv_json, self._router):
                self._dispatcher.dispatch(msg['action'], identifier, msg)
        for k in socks:
            if k in self._req_socks and socks[k] == zmq.POLLIN:
                for msg in utils.iter_recv(utils.req_recv_json, k):
                    self._dispatcher.dispatch(msg['action'], msg)

    def _on_ns_register_controller_req(self, identifier, msg):
        req_sock = self._context.socket(zmq.REQ)
        req_sock.connect('{}://{}:{}'.format(msg['ctl_protocal'], msg['ctl_addr'], msg['ctl_port']))
        self.storage.register(msg, req_sock)
        self._req_socks.add(req_sock)
        self._poller.register(req_sock, zmq.POLLIN)
        utils.router_send_json(self._router, identifier, {'action': _configs.Actions.NS_REGISTER_CTL_REP})
        logger.info('Controller registered: {}.'.format(msg['uid']))

    def _on_ns_register_outputs_req(self, identifier, msg):
        self.storage.register_outputs(msg)

        all_peers_to_inform = set()
        for i in msg['outputs']:
            all_peers_to_inform = all_peers_to_inform.union(self.storage.get_inputs(i))
        for peer in all_peers_to_inform:
            self._control_send_queue.put({
                'sock': self.storage.get_req_sock(peer),
                'countdown': _configs.CTL_CTL_SND_COUNTDOWN,
                'payload': {
                    'action': _configs.Actions.NS_NOTIFY_OPEN_REQ,
                    'uid': msg['uid'],
                    'info': self.storage.get(msg['uid'])
                },
            })

        utils.router_send_json(self._router, identifier, {'action': _configs.Actions.NS_REGISTER_OUTPUTS_REP})

        logger.info('Controller pipes registered: out={} (uid="{}").'.format(msg['outputs'], msg['uid']))

    def _on_ns_register_inputs_req(self, identifier, msg):
        self.storage.register_inputs(msg)

        res = {}
        for name in msg['inputs']:
            all_pipes = self.storage.get_outputs(name)
            all_pipes = list(map(self.storage.get, all_pipes))
            res[name] = all_pipes

        utils.router_send_json(self._router, identifier, {
            'action': _configs.Actions.NS_REGISTER_INPUTS_REP,
            'results': res
        })

    def _on_ns_heartbeat_req(self, identifier, msg):
        if self.storage.contains(msg['uid']):
            self.storage.get(msg['uid'])['last_heartbeat'] = time.time()
            logger.debug('Heartbeat {}: time={}.'.format(msg['uid'], time.time()))
            utils.router_send_json(self._router, identifier, {
                'action': _configs.Actions.NS_HEARTBEAT_REP
            })


def run_name_server(host=None, port=None, protocal=None):
    host = host or _configs.NS_CTL_HOST
    port = port or _configs.NS_CTL_PORT
    protocal = protocal or _configs.NS_CTL_PROTOCAL
    NameServer(host=host, port=port, protocal=protocal).mainloop()
