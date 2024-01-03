import amqwrapper
import time 
def message_reception (message): 
    print (message)

c=amqwrapper.Connection ("queue", message_reception, False)
time.sleep(10000)
