import requests

response = requests.post(url="http://127.0.0.1:8000/chat", data={"prompt": "https://www.ushareit.com/terms/"})
print(response)