import json
from collections import defaultdict
from gevent.subprocess import Popen, PIPE
import gevent


class TailFDispatcher(object):
    def __init__(self, subscribe_files):
        super(TailFDispatcher, self).__init__()
        self.subscribing_files = subscribe_files
        self.processes = []
        self.queues = []

        for file_path in subscribe_files:
            self.processes.append(gevent.spawn(self._tail, file_path))
        self.processes.append(gevent.spawn(self._avoid_timeout))

    def subscribe(self, queue):
        for file_path in self.subscribing_files:
            with open(file_path) as fp:
                def notify(msg):
                    queue.put(msg)

                for row in fp:
                    gevent.spawn(notify, row)
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

    def _tail(self, file_path):
        def notify(msg):
            for queue in self.queues[:]:
                queue.put(msg)

        p = Popen(['tail', '-n', '0', '-f', file_path], stdout=PIPE)
        self.processes.append(p)
        while True:
            gevent.spawn(notify, p.stdout.readline().strip())

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

    def file_subscribe(self, model_id, subscribe_files):
        model_id = int(model_id)
        self.tail_processes[model_id] = TailFDispatcher(subscribe_files)

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
