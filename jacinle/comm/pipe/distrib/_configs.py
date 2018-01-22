# -*- coding: utf-8 -*-
# File   : _configs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 22/01/2018
#
# This file is part of Jacinle.

NS_CTL_PROTOCAL = 'tcp'
NS_CTL_HOST = '*'
NS_CTL_PORT = '43521'
NS_HEARTBEAT_INTERVAL = 3
NS_CLEANUP_WAIT = 10

CTL_CTL_SND_COUNTDOWN = 5
CTL_CTL_HWM = 5
CTL_DAT_SND_COUNTDOWN = 5
CTL_DAT_HWM = 5

CTL_DAT_PROTOCAL = 'tcp'
CTL_DAT_HOST = '*'


class Actions:
    NS_REGISTER_CTL_REQ = 'ns-register-ctl-req'
    NS_REGISTER_CTL_REP = 'ns-register-ctl-rep'

    NS_REGISTER_PIPE_REQ = 'ns-register-pipe-req'
    NS_REGISTER_PIPE_REP = 'ns-register-pipe-rep'

    NS_HEARTBEAT_REQ = 'ns-heartbeat-req'
    NS_HEARTBEAT_REP = 'ns-heartbeat-rep'

    NS_QUERY_OPIPE_REQ = 'ns-query-opipe-req'
    NS_QUERY_OPIPE_REP = 'ns-query-opipe-rep'

    CTL_NOTIFY_OPEN_REQ = 'ctl-notify-open-req'
    CTL_NOTIFY_OPEN_REP = 'ctl-notify-open-rep'
    CTL_NOTIFY_CLOSE_REQ = 'ctl-notify-close-req'
    CTL_NOTIFY_CLOSE_REP = 'ctl-notify-close-rep'

    CTL_CONNECT_REQ = 'ctl-connect-req'
    CTL_CONNECT_REP = 'ctl-connect-rep'
    CTL_CONNECTED_REQ = 'ctl-connected-req'
    CTL_CONNECTED_REP = 'ctl-connected-rep'
