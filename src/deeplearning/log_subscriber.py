from collections import defaultdict
from gevent.subprocess import Popen
from gevent.queue import Channel
import gevent


class LogSubscriber(object):
    def __init__(self):
        super(LogSubscriber, self).__init__()
        self.tail_processes = {}
        self.channels = defaultdict(list)

    def tail(self, model_id, file_path):
        def notify(msg):
            for channel in self.channels[model_id]:
                channel.put(msg)

        p = Popen(['tail', '-n', '0', '-f', file_path])
        while True:
            gevent.spawn(notify, p.stdout.readline().strip())

    def file_subscribe(self, model_id, file_path):
        output = Channel()
        greenlet = gevent.spawn(self.tail, model_id, output, file_path)
        self.tail_processes[model_id] = greenlet

    def subscribe(self, model_id, channel):
        self.channels[model_id].append(channel)

    def unsubscribe(self, model_id, channel):
        self.channels[model_id].remove(channel)

    def terminate_train(self, model_id):
        if model_id in self.tail_processes:
            self.tail_processes[model_id].kill()
