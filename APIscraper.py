import json
import requests

api_token = '5ca98480-625f-4200-86b6-4eccd362b525'
api_url_base = 'https://content.guardianapis.com/us/commentisfree/'


headers = {'Content-Type': 'application/json',
           'Authorization': 'Bearer {0}'.format(api_token)}

def get_op_eds():
    api_url = "https://content.guardianapis.com/search?section=commentisfree&show-blocks=body&show-tags=contributor&api-key=5ca98480-625f-4200-86b6-4eccd362b525"
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        print(response.status_code)
        return None


op_eds = get_op_eds()
authcounts = {}

if op_eds is not None:
    print("Authors: ")
    n = True
    print(len(op_eds['response']['results']))
    for v in op_eds['response']['results']:
        if 'tags' in v and len(v['tags']) > 0:
            t = v['tags'][0]['webTitle']
            if t in authcounts:
                authcounts[t] += 1
            else:
                authcounts[t] = 0
for k, v in authcounts.items():
    print(k + ": " + str(v))
