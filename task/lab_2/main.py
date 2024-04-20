import time
from util.env_vars import config

if __name__ == '__main__':
    timestamp = str(time.time())
    
    print(config)
    print(timestamp)