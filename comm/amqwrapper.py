import stomp
import time 
class Listener (stomp.ConnectionListener):
    def __init__ (self, callback):
        self.callback=callback            
    def on_error(self, frame):
        print ('On error', frame.body)

    def on_message(self, frame ):
        print ('Message received')
        self.callback(frame.body)
class Connection :
    def __init__ (self, queue, callback, sender):
        try:
            hosts = [('localhost', 61613)]
            self.conn = stomp.Connection(host_and_ports=hosts)

            self.conn.set_listener(" ",Listener(callback))
            self.conn.connect('admin', 'admin', wait=True)# Register a consumer with ActiveMQ. This tells ActiveMQ to send all # messages received on the queue 'queue-1' to this listener
            self.queue_name = '/queue/'+queue
            self.sender=sender
            if (sender==False):
                self.conn.subscribe(destination=self.queue_name, id=1, ack='auto')
        except: 
            print ("Error opening connetion")
    def send (self, message):
        if (self.sender):
            self.conn.send(body=message , destination = self.queue_name)


