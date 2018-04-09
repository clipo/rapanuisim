import json
from pprint import pprint

with open('test-config.json') as data_file:    
    data = json.load(data_file)
#pprint(data)
for key in data.keys():
    print key
