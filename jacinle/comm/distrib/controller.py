#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import collections
import contextlib
import os
import queue
import random
import threading

import zmq

from jacinle.comm.distrib import _configs
from jacinle.concurrency import zmq_utils as utils
from jacinle.logging import get_logger
from jacinle.utils.registry import CallbackRegistry

logger = get_logger(__file__)

__all__ = [
    'Controller', 'control'
]


# mapping from pipe_name => list of pipe
class ControllerPipeStorage(collections.defaultdict):
    def __init__(self):
        super().__init__(list)

    def pipe_info(self):
        return [(k, p.identifier) for k, pipes in self.items() for p in pipes]

    def put(self, pipe):
        self[pipe.name].append(pipe)

    @staticmethod
    def fn_filter_notempty(pipe):
        return not pipe.empty()

    def filter_notempty(self, name):
        return list(filter(ControllerPipeStorage.fn_filter_notempty, self[name]))

    @staticmethod
    def fn_filter_notfull(pipe):
        return not pipe.full()

    def filter_notfull(self, name):
        return list(filter(ControllerPipeStorage.fn_filter_notfull, self[name]))


ControlMessage = collections.namedtuple('ControlMessage', ['sock', 'identifier', 'payload', 'countdown'])
ControllerPeer = collections.namedtuple('ControllerPeer', ['info', 'csock'])
PipePeer = collections.namedtuple('PipePeer', ['addr', 'port', 'dsock', 'pipes', 'ids'])

BroadcastMessage = collections.namedtuple('BroadcastMessage', ['from_identifier', 'payload'])
UnicastMessage = collections.namedtuple('UnicastMessage', ['from_identifier', 'to_identifier', 'payload'])


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

        # socket pools
        self._ns_socket = None

        # control router and dispatcher
        self._control_router = None
        self._control_router_port = 0
        self._control_dispatcher = CallbackRegistry()
        # queue of ControlMessage
        self._control_mqueue = queue.Queue()
        self._control_poller = zmq.Poller()
        # peers respect to the controller
        # map uid => ControllerPeer
        self._controller_peers = dict()

        # the peers of input pipes (i.e. the output pipes)
        self._input_from = dict()
        self._input_cache = dict()

        self._output_to = dict()
        self._output_to_pipe = collections.defaultdict(dict)  # the peers of output pipes
        self._output_to_id = dict()
        self._output_cache = dict()  # map pipe_name => cache

        self._data_poller = zmq.Poller()

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
                self._imanager.put(pipe)
            else:
                assert pipe.direction == 'OUT'
                self._omanager.put(pipe)
            pipe.set_controller(self)

        # setup ns socket
        self._ns_socket = self.socket(zmq.REQ)
        self._ns_socket.connect(os.getenv(
            'JAC_NAME_SERVER', '{}://localhost:{}'.format(
                _configs.NS_CTL_PROTOCAL, _configs.NS_CTL_PORT
            )
        ))
        self._control_poller.register(self._ns_socket, zmq.POLLIN)

        # setup router socket
        self._control_router = self.socket(zmq.ROUTER)
        self._control_router_port = self._control_router.bind_to_random_port('tcp://*')
        self._control_poller.register(self._control_router, zmq.POLLIN)

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
            'action': _configs.Actions.NS_REGISTER_OUTPUTS_REQ,
            'uid': self._uid,
            'outputs': list(self._omanager.keys())
        })
        assert response['action'] == _configs.Actions.NS_REGISTER_OUTPUTS_REP

        # query name-server for ipipes
        response = utils.req_send_and_recv(self._ns_socket, {
            'action': _configs.Actions.NS_REGISTER_INPUTS_REQ,
            'uid': self._uid,
            'inputs': list(self._imanager.keys())
        })
        assert response['action'] == _configs.Actions.NS_REGISTER_INPUTS_REP
        logger.info('IPipes query {}.'.format(response['results']))

        self._initialize_recv_peers(response['results'])

        # setup dispatcher
        self._control_dispatcher.register(_configs.Actions.NS_HEARTBEAT_REP, lambda msg: None)
        self._control_dispatcher.register(_configs.Actions.CTL_CONNECT_REQ, self._on_ctl_connect_req)
        self._control_dispatcher.register(_configs.Actions.CTL_CONNECT_REP, self._on_ctl_connect_rep)
        self._control_dispatcher.register(_configs.Actions.CTL_CONNECTED_REQ, self._on_ctl_connected_req)
        self._control_dispatcher.register(_configs.Actions.CTL_CONNECTED_REP, lambda msg: None)
        self._control_dispatcher.register(_configs.Actions.NS_NOTIFY_OPEN_REQ, self._on_ctl_notify_open_req)
        self._control_dispatcher.register(_configs.Actions.NS_NOTIFY_OPEN_REP, lambda msg: None)
        self._control_dispatcher.register(_configs.Actions.NS_NOTIFY_CLOSE_REQ, self._on_ctl_notify_close_req)
        self._control_dispatcher.register(_configs.Actions.NS_NOTIFY_CLOSE_REP, lambda msg: None)

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
            nr_done = 0

            socks = dict(self._control_poller.poll(0))
            nr_done += self._main_do_control_recv(socks)
            nr_done += self._main_do_control_send()

            socks = dict(self._data_poller.poll(0))
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
        for name in self._input_from:
            cache = self._input_cache.pop(name, None)
            if cache is None:
                peer = self._input_from[name]
                if peer.dsock in in_socks:
                    msg = utils.pull_pyobj(peer.dsock)
                    cache = (msg['name'], msg['from_identifier'], msg.get('to_identifier', None), msg['data'])

            if cache is None:
                continue

            nr_done_this = 0
            if cache[2] is None:  # is broadcast
                for p in self._imanager.filter_notfull(cache[0]):
                    p.raw_queue.put_nowait(BroadcastMessage(cache[1], cache[-1]))
                    nr_done_this += 1
            else:
                for p in self._imanager.filter_notfull(cache[0]):
                    if p.identifier == cache[2]:
                        p.raw_queue.put_nowait(UnicastMessage(cache[1], cache[2], cache[-1]))
                        nr_done_this += 1

            if nr_done_this > 0:
                nr_done += nr_done_this
            else:
                self._input_cache[name] = cache

        return nr_done

    def _main_do_data_send(self):
        nr_done = 0
        for name in self._omanager.keys():
            cache = self._output_cache.get(name, None)

            if cache is None:
                pipes = self._omanager.filter_notempty(name)
                if len(pipes) != 0:
                    pipe = random.choice(pipes)
                    cache = pipe.raw_queue.get_nowait()
                    self._output_cache[name] = cache

            if cache is None:
                continue

            nr_done_this = 0

            if isinstance(cache, BroadcastMessage):
                for peer in self._output_to_pipe[name].values():
                    nr_done_this += utils.push_pyobj(peer.dsock, {
                        'uid': self._uid,
                        'name': name,
                        'from_identifier': cache.from_identifier,
                        'data': cache.payload
                    }, flag=zmq.NOBLOCK)
            elif isinstance(cache, UnicastMessage):
                if (name, cache.to_identifier) in self._output_to_id:
                    peer = self._output_to_id[(name, cache.to_identifier)]
                    nr_done_this += utils.push_pyobj(peer.dsock, {
                        'uid': self._uid,
                        'name': name,
                        'from_identifier': cache.from_identifier,
                        'to_identifier': cache.to_identifier,
                        'data': cache.payload
                    }, flag=zmq.NOBLOCK)
            else:
                raise TypeError('Unknown message type: {}.'.format(type(cache)))

            if nr_done_this > 0:
                self._output_cache[name] = None

            nr_done += nr_done_this

        return nr_done

    # BEGIN:: Connection

    def _initialize_recv_peers(self, results):
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
        self._control_poller.register(sock, zmq.POLLIN)
        self._control_mqueue.put(ControlMessage(sock, None, {
            'action': _configs.Actions.CTL_CONNECT_REQ,
            'uid': self._uid,
            'inputs': self._imanager.pipe_info()
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))
        self._controller_peers[uid] = ControllerPeer(info, sock)
        logger.info('Connecting to "{}".'.format(uid))

    def _on_ctl_connect_req(self, identifier, msg):
        uid, pipes = msg['uid'], msg['inputs']

        flag = False
        for name, _ in pipes:
            if name in self._omanager:
                flag = True
                break

        if flag:
            response = {}
            if uid in self._output_to:
                port = self._output_to[uid].port
            else:
                sock = self.socket(zmq.PUSH)
                port = sock.bind_to_random_port('{}://{}'.format(_configs.CTL_DAT_PROTOCAL, _configs.CTL_DAT_HOST))
                pipes_rec = {p[0] for p in pipes}
                ids_rec = {tuple(p) for p in pipes}
                peer = PipePeer(self._addr, port, sock, pipes_rec, ids_rec)

                self._output_to[uid] = peer
                for p in pipes_rec:
                    self._output_to_pipe[p][uid] = peer
                for i in ids_rec:
                    self._output_to_id[i] = peer

                logger.info('Connection opened for "{}": port={}.'.format(uid, port))

            if port > 0:
                response = {
                    'dat_protocal': _configs.CTL_DAT_PROTOCAL,
                    'dat_addr': self._addr,
                    'dat_port': port
                }

            self._control_mqueue.put(ControlMessage(self._control_router, identifier, {
                'action': _configs.Actions.CTL_CONNECT_REP,
                'uid': self._uid,
                'conn': response
            }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))

    def _on_ctl_connect_rep(self, msg):
        uid, conn = msg['uid'], msg['conn']

        if len(conn) and uid not in self._input_from:
            sock = self.socket(zmq.PULL)
            sock.connect('{}://{}:{}'.format(conn['dat_protocal'], conn['dat_addr'], conn['dat_port']))
            self._data_poller.register(sock, zmq.POLLIN)
            self._input_from[uid] = PipePeer(conn['dat_addr'], conn['dat_port'], sock, None, None)
            logger.info('Connection established to "{}": remote_port={}.'.format(uid, conn['dat_port']))

        self._control_mqueue.put(ControlMessage(self._controller_peers[uid].csock, None, {
            'action': _configs.Actions.CTL_CONNECTED_REQ,
            'uid': self._uid
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))

    def _on_ctl_connected_req(self, identifier, msg):
        self._control_mqueue.put(ControlMessage(self._control_router, identifier, {
            'action': _configs.Actions.CTL_CONNECTED_REP,
            'uid': self._uid
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))
        logger.info('Connection established for "{}".'.format(msg['uid']))

    # END:: Connection

    def _on_ctl_notify_open_req(self, identifier, msg):
        uid = msg['uid']
        if uid not in self._controller_peers:
            self._controller_peers[uid] = ControllerPeer(msg['info'], None)
            self._do_setup_ctl_peer(uid)
        self._control_mqueue.put(ControlMessage(self._control_router, identifier, {
            'action': _configs.Actions.NS_NOTIFY_OPEN_REP,
            'uid': self._uid
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))
        logger.info('Found new controller: "{}".'.format(uid))

    def _on_ctl_notify_close_req(self, identifier, msg):
        uid = msg['uid']
        if uid in self._controller_peers:
            peer = self._controller_peers.pop(uid)
            self._control_poller.unregister(peer.csock)
            self.close_socket(peer.csock)
        if uid in self._input_from:
            peer = self._input_from.pop(uid)
            self._data_poller.unregister(peer.dsock)
            self.close_socket(peer.dsock)
        if uid in self._input_cache:
            del self._input_cache[uid]
        if uid in self._output_to:
            peer = self._output_to.pop(uid)
            self.close_socket(peer.dsock)
            for k in peer.pipes:
                self._output_to_pipe[k].pop(uid)
            for k in peer.ids:
                self._output_to_id.pop(k)
        if uid in self._output_cache:
            del self._output_cache[uid]
        self._control_mqueue.put(ControlMessage(self._control_router, identifier, {
            'action': _configs.Actions.NS_NOTIFY_CLOSE_REP,
            'uid': self._uid
        }, countdown=_configs.CTL_CTL_SND_COUNTDOWN))
        logger.info('Close timeout controller: "{}".'.format(uid))


@contextlib.contextmanager
def control(pipes):
    ctl = Controller()
    ctl.initialize(pipes)
    yield ctl
    ctl.finalize()
