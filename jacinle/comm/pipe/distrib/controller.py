# -*- coding: utf-8 -*-
# File   : controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 22/01/2018
#
# This file is part of Jacinle.

import zmq
import os
import collections
import threading
import queue
import contextlib
import random

from jacinle.logging import get_logger
from jacinle.utils.registry import CallbackRegistry

from . import _configs
from ... import zmq_utils as utils

logger = get_logger(__file__)

__all__ = [
    'Controller', 'control'
]


class ControllerPipeStorage(object):
    def __init__(self):
        # mapping from pipe_name => list of pipes
        self._pipes = {}

    @property
    def names(self):
        return self._pipes.keys()

    def has_pipes(self, name):
        return len(self._pipes[name]) > 0 if name in self._pipes else False

    def get_pipes(self, name):
        return self._pipes.get(name, [])

    def put_pipe(self, pipe):
        self._pipes.setdefault(pipe.name, []).append(pipe)
        return self

    @staticmethod
    def fn_filter_notempty(pipe):
        return not pipe.empty()

    def filter_notempty(self, name):
        return list(filter(ControllerPipeStorage.fn_filter_notempty, self.get_pipes(name)))

    @staticmethod
    def fn_filter_notfull(pipe):
        return not pipe.full()

    def filter_notfull(self, name):
        return list(filter(ControllerPipeStorage.fn_filter_notfull, self.get_pipes(name)))


ControlMessage = collections.namedtuple('ControlMessage', ['sock', 'identifier', 'payload', 'countdown'])
ControllerPeer = collections.namedtuple('ControllerPeer', ['info', 'csock'])
PipePeer = collections.namedtuple('PipePeer', ['addr', 'port', 'dsock'])


class Controller(object):
    def __init__(self):
        self._uid = utils.uid()
        self._addr = utils.get_addr()

        # pipes of this controller
        self._imanager = ControllerPipeStorage()
        self._omanager = ControllerPipeStorage()

        # context and poller
        self._context = zmq.Context()
        self._context.sndhwm = _configs.CTL_CTL_HWM
        self._context.rcvhwm = _configs.CTL_CTL_HWM
        self._poller = zmq.Poller()

        # socket pools
        self._ns_socket = None

        # control router and dispatcher
        self._control_router = None
        self._control_router_port = 0
        self._control_dispatcher = CallbackRegistry()
        # queue of ControlMessage
        self._control_mqueue = queue.Queue()

        # peers respect to the controller
        # map uid => ContorllerPeer
        self._controller_peers = dict()
        # the peers of input pipes (i.e. the output pipes)
        # map pipe_name => dict of <uid, PipePeer>
        self._ipipe_peers = dict()
        # the peers of output pipes
        # map pipe_name => dict of <uid, PipePeer>
        self._opipe_peers = dict()
        # map pipe_name => cache
        self._opipe_cache = dict()

        # threads and stop-event
        self._all_socks = set()
        self._all_threads = []
        self._stop_event = threading.Event()

    def socket(self, socket_type):
        sock = self._context.socket(socket_type)
        self._all_socks.add(sock)
        return sock

    def close_socket(self, sock):
        utils.graceful_close(sock)
        self._all_socks.remove(sock)
        return self

    def initialize(self, pipes=None):
        pipes = pipes or []

        for pipe in pipes:
            if pipe.direction == 'IN':
                self._imanager.put_pipe(pipe)
            else:
                assert pipe.direction == 'OUT'
                self._omanager.put_pipe(pipe)
            pipe.set_controller(self)

        # setup ns socket
        self._ns_socket = self.socket(zmq.REQ)
        self._ns_socket.connect(os.getenv(
            'TART_NAME_SERVER', '{}://localhost:{}'.format(
                _configs.NS_CTL_PROTOCAL, _configs.NS_CTL_PORT
            )
        ))
        self._poller.register(self._ns_socket, zmq.POLLIN)

        # setup router socket
        self._control_router = self.socket(zmq.ROUTER)
        self._control_router_port = self._control_router.bind_to_random_port('tcp://*')
        self._poller.register(self._control_router, zmq.POLLIN)

        # register on the name-server
        response = utils.req_send_and_recv(self._ns_socket, {
            'action': _configs.Actions.NS_REGISTER_CTL_REQ,
            'uid': self._uid,
            'ctl_protocal': 'tcp',
            'ctl_addr': self._addr,
            'ctl_port': self._control_router_port,
            'meta': {}
        })
        assert response['action'] == _configs.Actions.NS_REGISTER_CTL_REP

        # register pipes on name-server
        response = utils.req_send_and_recv(self._ns_socket, {
            'action': _configs.Actions.NS_REGISTER_PIPE_REQ,
            'uid': self._uid,
            'ipipes': list(self._imanager.names),
            'opipes': list(self._omanager.names)
        })
        assert response['action'] == _configs.Actions.NS_REGISTER_PIPE_REP

        # query name-server for ipipes
        response = utils.req_send_and_recv(self._ns_socket, {
            'action': _configs.Actions.NS_QUERY_OPIPE_REQ,
            'ipipes': list(self._imanager.names)
        })
        assert response['action'] == _configs.Actions.NS_QUERY_OPIPE_REP
        logger.info('IPipes query {}.'.format(response['results']))

        self._initialize_ipipe_peers(response['results'])

        # setup dispatcher
        self._control_dispatcher.register(_configs.Actions.CTL_CONNECT_REP, self._on_ctl_connect_rep)
        self._control_dispatcher.register(_configs.Actions.CTL_CONNECT_REQ, self._on_ctl_connect_req)
        self._control_dispatcher.register(_configs.Actions.CTL_CONNECTED_REQ, self._on_ctl_connected_req)
        self._control_dispatcher.register(_configs.Actions.CTL_NOTIFY_OPEN_REQ, self._on_ctl_notify_open_req)
        self._control_dispatcher.register(_configs.Actions.CTL_NOTIFY_CLOSE_REQ, self._on_ctl_notify_close_req)

        # run threads
        self._all_threads.append(threading.Thread(target=self._main, name='ctl-main'))
        self._all_threads.append(threading.Thread(target=self._main_heartbeat, name='ctl-main-ns-heartbeat'))
        for i in self._all_threads:
            i.start()

    def finalize(self):
        self._stop_event.set()
        for i in self._all_threads:
            i.join()
        for sock in self._all_socks:
            utils.graceful_close(sock)

    def _main(self):
        wait = 0
        while True:
            if self._stop_event.wait(wait / 1000):
                break

            socks = dict(self._poller.poll(0))

            nr_done = 0
            nr_done += self._main_do_control_recv(socks)
            nr_done += self._main_do_control_send()
            nr_done += self._main_do_data_recv(socks)
            nr_done += self._main_do_data_send()

            if nr_done > 0:
                wait = wait / 2 if wait > 1 else 0
            else:
                wait = wait + 1 if wait < 50 else 50

    def _main_heartbeat(self):
        while True:
            self._control_mqueue.put(ControlMessage(self._ns_socket, None, {
                'action': _configs.Actions.NS_HEARTBEAT_REQ,
                'uid': self._uid
            }, countdown=0))

            if self._stop_event.wait(_configs.NS_HEARTBEAT_INTERVAL):
                break

    def _main_do_control_recv(self, socks):
        nr_done = 0

        # ns
        if self._ns_socket in socks:
            for msg in utils.iter_recv(utils.req_recv_json, self._ns_socket):
                self._control_dispatcher.dispatch(msg['action'], msg)
                nr_done += 1

        # router
        if self._control_router in socks:
            for identifier, msg in utils.iter_recv(utils.router_recv_json, self._control_router):
                self._control_dispatcher.dispatch(msg['action'], identifier, msg)
                nr_done += 1

        for info, csock in self._controller_peers.values():
            if csock in socks:
                for msg in utils.iter_recv(utils.req_recv_json, csock):
                    self._control_dispatcher.dispatch(msg['action'], msg)
                    nr_done += 1

        return nr_done

    def _main_do_control_send(self):
        nr_scheduled = self._control_mqueue.qsize()
        nr_done = 0
        for i in range(nr_scheduled):
            job = self._control_mqueue.get()
            if job.identifier is not None:
                rc = utils.router_send_json(job.sock, job.identifier, job.payload, flag=zmq.NOBLOCK)
            else:
                rc = utils.req_send_json(job.sock, job.payload, flag=zmq.NOBLOCK)
            if not rc:
                if job.countdown > 0:
                    self._control_mqueue.put(ControlMessage(job[0], job[1], job[2], job.countdown - 1))
            else:
                nr_done += 1

        return nr_done

    def _main_do_data_recv(self, in_socks):
        nr_done = 0
        for name in self._imanager.names:
            pipes = self._imanager.filter_notfull(name)
            if len(pipes) == 0:
                continue

            socks= []
            for peer in self._ipipe_peers.get(name, {}).values():
                if peer.dsock in in_socks:
                    socks.append(peer.dsock)
            if len(socks) == 0:
                continue

            sock = random.choice(socks)
            msg = utils.pull_pyobj(sock)
            if msg is not None:
                for p in pipes:
                    p.put_nowait(msg['data'])
                    nr_done += 1

        return nr_done

    def _main_do_data_send(self):
        nr_done = 0
        for name in self._omanager.names:
            cache = self._opipe_cache.get(name, None)

            if cache is None:
                pipes = self._omanager.filter_notempty(name)
                if len(pipes) != 0:
                    pipe = random.choice(pipes)
                    cache = pipe.get_nowait()
                    self._opipe_cache[name] = cache

            if cache is None:
                continue

            nr_done_this = 0
            for peer in self._opipe_peers.get(name, {}).values():
                nr_done_this += utils.push_pyobj(peer.dsock, {
                    'uid': self._uid,
                    'data': cache
                }, flag=zmq.NOBLOCK)

            if nr_done_this > 0:
                self._opipe_cache[name] = None

            nr_done += nr_done_this

        return nr_done

    # BEGIN:: Connection

    def _initialize_ipipe_peers(self, results):
        for peers in results.values():
            for info in peers:
                uid = info['uid']
                if uid not in self._controller_peers:
                    self._controller_peers[uid] = ControllerPeer(info, None)
                    self._do_setup_ctl_peer(uid)

    def _do_setup_ctl_peer(self, uid):
        info, sock = self._controller_peers[uid]
        if sock is not None:
            return

        sock = self.socket(zmq.REQ)
        sock.connect('{}://{}:{}'.format(info['ctl_protocal'], info['ctl_addr'], info['ctl_port']))
        self._poller.register(sock, zmq.POLLIN)
        self._control_mqueue.put(ControlMessage(sock, None, {
            'action': _configs.Actions.CTL_CONNECT_REQ,
            'uid': self._uid,
            'ipipes': list(self._imanager.names)
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))
        self._controller_peers[uid] = ControllerPeer(info, sock)
        logger.info('Connecting to {}.'.format(uid))

    def _on_ctl_connect_req(self, identifier, msg):
        uid, pipes = msg['uid'], msg['ipipes']
        response = {}
        for name in pipes:
            if not self._omanager.has_pipes(name):
                continue

            peers = self._opipe_peers.setdefault(name, {})
            if uid in peers:
                port = peers['uid'].port
            else:
                sock = self.socket(zmq.PUSH)
                port = sock.bind_to_random_port('{}://{}'.format(_configs.CTL_DAT_PROTOCAL, _configs.CTL_DAT_HOST))
                peers[uid] = PipePeer(self._addr, port, sock)
                logger.info('Connection opened for {}: pipe={}, port={}.'.format(uid, name, port))

            if port > 0:
                response[name] = {
                    'dat_protocal': _configs.CTL_DAT_PROTOCAL,
                    'dat_addr': self._addr,
                    'dat_port': port
                }

        self._control_mqueue.put(ControlMessage(self._control_router, identifier, {
            'action': _configs.Actions.CTL_CONNECT_REP,
            'uid': self._uid,
            'opipes': response
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))

    def _on_ctl_connect_rep(self, msg):
        uid, pipes = msg['uid'], msg['opipes']

        for name, info in pipes.items():
            sock = self.socket(zmq.PULL)
            sock.connect('{}://{}:{}'.format(info['dat_protocal'], info['dat_addr'], info['dat_port']))
            self._poller.register(sock, zmq.POLLIN)
            self._ipipe_peers.setdefault(name, {})[uid] = PipePeer(info['dat_addr'], info['dat_port'], sock)
            logger.info('Connection established to {}: pipe={}, remote_port={}.'.format(uid, name, info['dat_port']))

        self._control_mqueue.put(ControlMessage(self._controller_peers[uid].csock, None, {
            'action': _configs.Actions.CTL_CONNECTED_REQ,
            'uid': self._uid
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))

    def _on_ctl_connected_req(self, identifier, msg):
        self._control_mqueue.put(ControlMessage(self._control_router, identifier, {
            'action': _configs.Actions.CTL_CONNECTED_REP,
            'uid': self._uid
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))
        logger.info('Connection established for {}.'.format(msg['uid']))

    # END:: Connection

    def _on_ctl_notify_open_req(self, identifier, msg):
        uid = msg['uid']
        if uid not in self._controller_peers:
            self._controller_peers[uid] = ControllerPeer(msg['info'], None)
            self._do_setup_ctl_peer(uid)
        self._control_mqueue.put(ControlMessage(self._control_router, identifier, {
            'action': _configs.Actions.CTL_NOTIFY_OPEN_REP,
            'uid': self._uid
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))
        logger.info('Found new controller {}.'.format(uid))

    def _on_ctl_notify_close_req(self, identifier, msg):
        uid = msg['uid']
        if uid in self._controller_peers:
            peer = self._controller_peers.pop(uid)
            self._poller.unregister(peer.csock)
            self.close_socket(peer.csock)
        for peers in self._ipipe_peers.values():
            if uid in peers:
                peer = peers.pop(uid)
                self._poller.unregister(peer.dsock)
                self.close_socket(peer.dsock)
        for peers in self._opipe_peers.values():
            if uid in peers:
                peer = peers.pop(uid)
                self.close_socket(peer.dsock)
        self._control_mqueue.put(ControlMessage(self._control_router, identifier, {
            'action': _configs.Actions.CTL_NOTIFY_CLOSE_REP,
            'uid': self._uid
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))
        logger.info('Close timeout controller {}.'.format(uid))


@contextlib.contextmanager
def control(pipes):
    ctl = Controller()
    ctl.initialize(pipes)
    yield ctl
    ctl.finalize()
