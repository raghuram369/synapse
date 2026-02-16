# Terminal 1: start Synapse
# $ synapse up --port 9470
# Terminal 2: any HTTP client
import requests
r = requests.post('http://localhost:9470/tool', json={'tool': 'remember', 'args': {'content': 'User likes jazz'}})
