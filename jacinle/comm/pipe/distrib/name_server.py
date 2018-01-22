# -*- coding: utf-8 -*-
# File   : name_server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 22/01/2018
#
# This file is part of Jacinle.

import zmq
import time
import threading
import queue

from jacinle.logging import get_logger
from jacinle.utils.registry import CallbackRegistry

from . import _configs
from ... import zmq_utils as utils

logger = get_logger(__file__)

__all__ = ['NameServer', 'run_name_server']


class NameServerControllerStorage(object):
    def __init__(self):
        self._all_peers = {}
        self._all_peers_req = {}
        self._ipipes = {}
        self._opipes = {}

    def register(self, info, req_sock):
        identifier = info['uid']
        assert identifier not in self._all_peers
        self._all_peers[identifier] = {
            'uid': info['uid'],
            'ctl_protocal': info['ctl_protocal'],
            'ctl_addr': info['ctl_addr'],
            'ctl_port': info['ctl_port'],
            'meta': info.get('meta', {}),
            'ipipes': [],
            'opipes': [],
            'last_heartbeat': time.time()
        }
        self._all_peers_req[identifier] = req_sock

    def register_pipes(self, info):
        controller = info['uid']
        assert controller in self._all_peers
        record = self._all_peers[controller]
        for i in record['ipipes']:
            self._ipipes.get(i, []).remove(controller)
        for i in record['opipes']:
            self._opipes.get(i, []).remove(controller)
        record['ipipes'] = info['ipipes']
        record['opipes'] = info['opipes']
        for i in record['ipipes']:
            self._ipipes.setdefault(i, []).append(controller)
        for i in record['opipes']:
            self._opipes.setdefault(i, []).append(controller)

    def unregister(self, identifier):
        if identifier in self._all_peers:
            info = self._all_peers.pop(identifier)
            for i in info['ipipes']:
                self._ipipes.get(i, []).remove(identifier)
            for i in info['opipes']:
                self._opipes.get(i, []).remove(identifier)
            return info, self._all_peers_req.pop(identifier)
        return None

    def get(self, identifier):
        return self._all_peers.get(identifier, None)

    def get_req_sock(self, identifier):
        return self._all_peers_req.get(identifier, None)

    def get_ipipe(self, name):
        return self._ipipes.get(name, [])

    def get_opipe(self, name):
        return self._opipes.get(name, [])

    def contains(self, identifier):
        return identifier in self._all_peers

    def all(self):
        return list(self._all_peers.keys())


class NameServer(object):
    def __init__(self, host=_configs.NS_CTL_HOST, port=_configs.NS_CTL_PORT, protocal=_configs.NS_CTL_PROTOCAL):
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
        self._dispatcher.register(_configs.Actions.NS_REGISTER_PIPE_REQ, self._on_ns_register_pipe_req)
        self._dispatcher.register(_configs.Actions.NS_QUERY_OPIPE_REQ, self._on_ns_query_opipe_req)
        self._dispatcher.register(_configs.Actions.NS_HEARTBEAT_REQ, self._on_ns_heartbeat_req)
        self._dispatcher.register(_configs.Actions.CTL_NOTIFY_OPEN_REP, lambda msg: None)
        self._dispatcher.register(_configs.Actions.CTL_NOTIFY_CLOSE_REP, lambda msg: None)

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
                for k in self.storage.all():
                    v = self.storage.get(k)

                    if (now - v['last_heartbeat']) > _configs.NS_CLEANUP_WAIT:
                        info, req_sock = self.storage.unregister(k)
                        self._poller.unregister(req_sock)
                        utils.graceful_close(req_sock)
                        self._req_socks.remove(req_sock)

                        # TODO:: use controller's heartbeat
                        all_peers_to_inform = set()
                        for i in info['ipipes']:
                            for j in self.storage.get_opipe(i):
                                all_peers_to_inform.add(j)
                        for i in info['opipes']:
                            for j in self.storage.get_ipipe(i):
                                all_peers_to_inform.add(j)
                        print('inform', all_peers_to_inform)

                        for peer in all_peers_to_inform:
                            self._control_send_queue.put({
                                'sock': self.storage.get_req_sock(peer),
                                'countdown': _configs.CTL_CTL_SND_COUNTDOWN,
                                'payload': {
                                    'action': _configs.Actions.CTL_NOTIFY_CLOSE_REQ,
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
                    print('drop job: ', job)

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

    def _on_ns_register_pipe_req(self, identifier, msg):
        self.storage.register_pipes(msg)

        all_peers_to_inform = set()
        for i in msg['opipes']:
            for j in self.storage.get_ipipe(i):
                all_peers_to_inform.add(j)
        print('inform', all_peers_to_inform)
        for peer in all_peers_to_inform:
            self._control_send_queue.put({
                'sock': self.storage.get_req_sock(peer),
                'countdown': _configs.CTL_CTL_SND_COUNTDOWN,
                'payload': {
                    'action': _configs.Actions.CTL_NOTIFY_OPEN_REQ,
                    'uid': msg['uid'],
                    'info': self.storage.get(msg['uid'])
                },
            })
        utils.router_send_json(self._router, identifier, {'action': _configs.Actions.NS_REGISTER_PIPE_REP})

        logger.info('Controller pipes registered: in={}, out={} (controller-uid={}).'.format(
            msg['ipipes'], msg['opipes'], msg['uid']))

    def _on_ns_query_opipe_req(self, identifier, msg):
        res = {}
        for name in msg['ipipes']:
            all_pipes = self.storage.get_opipe(name)
            all_pipes = list(map(self.storage.get, all_pipes))
            res[name] = all_pipes

        utils.router_send_json(self._router, identifier, {
            'action': _configs.Actions.NS_QUERY_OPIPE_REP,
            'results': res
        })

    def _on_ns_heartbeat_req(self, identifier, msg):
        if self.storage.contains(msg['uid']):
            self.storage.get(msg['uid'])['last_heartbeat'] = time.time()
            print('Heartbeat {}: time={}'.format(msg['uid'], time.time()))
            utils.router_send_json(self._router, identifier, {
                'action': _configs.Actions.NS_HEARTBEAT_REP
            })


def run_name_server(host=None, port=None, protocal=None):
    host = host or _configs.NS_CTL_HOST
    port = port or _configs.NS_CTL_PORT
    protocal = protocal or _configs.NS_CTL_PROTOCAL
    NameServer(host=host, port=port, protocal=protocal).mainloop()
