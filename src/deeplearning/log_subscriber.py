import json
from collections import defaultdict
from gevent.subprocess import Popen, PIPE
import gevent


class LogSubscriber(object):
    def __init__(self):
        super(LogSubscriber, self).__init__()
        self.subscribing_files = {}
        self.tail_processes = {}
        self.queues = defaultdict(list)

    def file_subscribe(self, model_id, file_path):
        model_id = int(model_id)
        self.subscribing_files[model_id] = file_path
        tail = gevent.spawn(self._tail, model_id, file_path)
        avoid_timeout = gevent.spawn(self._avoid_timeout, model_id)
        self.tail_processes[model_id] = [tail, avoid_timeout]

    def subscribe(self, model_id, queue):
        model_id = int(model_id)
        if model_id in self.subscribing_files:
            with open(self.subscribing_files[model_id]) as fp:
                def notify(msg):
                    queue.put(msg)

                for row in fp:
                    gevent.spawn(notify, row)
        self.queues[model_id].append(queue)

    def unsubscribe(self, model_id, queue):
        model_id = int(model_id)
        self.queues[model_id].remove(queue)

    def terminate_train(self, model_id):
        model_id = int(model_id)
        if model_id in self.tail_processes:
            for process in self.tail_processes[model_id]:
                process.kill()

            def notify():
                for queue in self.queues[model_id][:]:
                    queue.put(json.dumps({
                        'type': 'end'
                    }))

            gevent.spawn(notify)
            del self.tail_processes[model_id]
            del self.subscribing_files[model_id]
            del self.queues[model_id]

    def _tail(self, model_id, file_path):
        def notify(msg):
            for queue in self.queues[model_id][:]:
                queue.put(msg)

        p = Popen(['tail', '-n', '0', '-f', file_path], stdout=PIPE)
        self.tail_processes[model_id].append(p)
        while True:
            gevent.spawn(notify, p.stdout.readline().strip())

    def _avoid_timeout(self, model_id):
        def notify():
            for queue in self.queues[model_id][:]:
                queue.put(None)

        while True:
            gevent.spawn(notify)
            gevent.sleep(45)


train_logger = LogSubscriber()
