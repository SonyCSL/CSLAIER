import json
from collections import defaultdict
from gevent.subprocess import Popen, PIPE
import gevent

TRAIN_LOG = 'log'
LINE_GRAPH = 'graph'


def wrap_log_type(row, log_type):
    return json.dumps({'type': log_type, 'data': row})


class TailFDispatcher(object):
    def __init__(self, train_log, line_graph):
        super(TailFDispatcher, self).__init__()
        self.train_log = train_log
        self.line_graph = line_graph
        self.processes = []
        self.queues = []

        if train_log:
            self.processes.append(gevent.spawn(self._tail, train_log, TRAIN_LOG))
        if line_graph:
            self.processes.append(gevent.spawn(self._tail, line_graph, LINE_GRAPH))

        self.processes.append(gevent.spawn(self._avoid_timeout))

    def subscribe(self, queue):
        def notify(msg):
            queue.put(msg)

        with open(self.train_log) as fp:
            for row in fp:
                gevent.spawn(notify, wrap_log_type(row, TRAIN_LOG))
        with open(self.line_graph) as fp:
            for row in fp:
                gevent.spawn(notify, wrap_log_type(row, LINE_GRAPH))

        self.queues.append(queue)

    def unsubscribe(self, queue):
        self.queues.remove(queue)

    def terminate(self):
        for process in self.processes:
            process.kill()

        def notify():
            for queue in self.queues[:]:
                queue.put(json.dumps({
                    'type': 'end'
                }))

        gevent.spawn(notify)

    def _tail(self, file_path, log_type):
        def notify(msg):
            for queue in self.queues[:]:
                queue.put(msg)

        p = Popen(['tail', '-n', '0', '-f', file_path], stdout=PIPE)
        self.processes.append(p)
        while True:
            row = p.stdout.readline()
            gevent.spawn(notify, wrap_log_type(row, log_type))

    def _avoid_timeout(self):
        def notify():
            for queue in self.queues[:]:
                queue.put(None)

        while True:
            gevent.spawn(notify)
            gevent.sleep(45)


class LogSubscriber(object):
    def __init__(self):
        super(LogSubscriber, self).__init__()
        self.tail_processes = {}

    def file_subscribe(self, model_id, train_log, line_graph):
        model_id = int(model_id)
        self.tail_processes[model_id] = TailFDispatcher(train_log, line_graph)

    def subscribe(self, model_id, queue):
        model_id = int(model_id)
        if model_id in self.tail_processes:
            self.tail_processes[model_id].subscribe(queue)

    def unsubscribe(self, model_id, queue):
        model_id = int(model_id)
        if model_id in self.tail_processes:
            self.tail_processes[model_id].unsubscribe(queue)

    def terminate_train(self, model_id):
        model_id = int(model_id)
        if model_id in self.tail_processes:
            self.tail_processes[model_id].terminate()


train_logger = LogSubscriber()
