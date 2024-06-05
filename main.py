import time
from flask import jsonify
import requests
import openai
import yaml
from SREChain import SREChain
from botocore.exceptions import NoCredentialsError, ClientError



def get_messages(auth):
    url = 'https://api.gluegroups.com/graphql'
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://api.gluegroups.com',
        'Authorization': f'Bearer {auth}'
    }
    data = {
        "query": """
        query GetThread($id: ID!) {
          node(id: $id) {
            id
            ... on Thread {
              subject
              messages(last: 10) {
                edges {
                  node {
                    text
                    user {
                      name
                    }
                  }
                }
              }
            }
          }
        }
        """,
        "variables": {"id": "thr_2hR4Bc7Qwjj4fXKrVkWXi7vg3ta"}
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    messages_data = response.json()
    messages = []
    for edge in messages_data['data']['node']['messages']['edges']:
        user_name = edge['node']['user']['name']
        text = edge['node']['text']
        messages.append(f"{user_name}: {text}")
    
    return "\n".join(messages)

def compose_reply_no_migrations(messages, problem_service, key):
    client = openai.OpenAI(api_key=key)

    prompt = f"""
    You are a site reliability engineer responding to a developers' message. The developers reported issues with the "{problem_service}" service. Based on the given messages, explain that no recent migrations affected this service and suggest troubleshooting alternatives.

    Messages:
    {messages}

    Reply:
    """

    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )

    reply = completion.choices[0].message.content.strip()
    return reply

def compose_reply_with_migrations(messages, problem_service, key, migrated_objects):
    client = openai.OpenAI(api_key=key)

    prompt = f"""
    You are a site reliability engineer responding to a developers' message. The developers reported issues with the "{problem_service}" service. Based on the given messages, explain which objects were migrated for which service, indicate that you have made the change, and ask them to let you know if the issue is resolved.

    Messages:
    {messages}

    Migrated Objects:
    {migrated_objects}

    Reply:
    """

    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )

    reply = completion.choices[0].message.content.strip()
    return reply

def send_reply(reply_text, auth):
    url = 'https://api.gluegroups.com/graphql'
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://api.gluegroups.com',
        'Authorization': f'Bearer {auth}'
    }
    data = {
        "query": """
        mutation SendMessage($input: SendMessageInput!) {
          sendMessage(input: $input) {
            message {
              id
              text
              createdAt
              user {
                id
                name
              }
            }
          }
        }
        """,
        "variables": {
            "input": {
                "threadID": "thr_2hR4Bc7Qwjj4fXKrVkWXi7vg3ta",
                "message": {
                    "text": reply_text,
                    "attachments": []
                }
            }
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()


def main():
    # Create an instance of SREChain
    sre_chain = SREChain()
    service_keys_str = []
    with open('config.yaml', 'r') as file:
      config = yaml.safe_load(file)
    
    openai_config = config['openai']

    glue_config = config['glue']

    curr_messages = []

    while(True):

      try:
          messages = get_messages(glue_config['auth'])
          if messages == curr_messages:
              continue
      except requests.exceptions.RequestException as e:
          print(f"Failed to get messages: {e}")
          return

      # Run the SREChain and get the response
      response = sre_chain.run_chain(messages)
      print("SREChain Response:", response)
      response = None

      try:
          services_response = requests.post(
              "http://127.0.0.1:5000/list",
              json={"filter_by_migrations": response}
          )
          services_response.raise_for_status()
          services_response_json = services_response.json()
          print("Services Response:", services_response_json)

          if services_response_json:
              for item in services_response_json:
                  key = item['key']
                  platform = item['platform']
                  migrate_response = requests.post(
                      "http://127.0.0.1:5000/migrate",
                      json={
                          "key": key,
                          "platform": platform
                      }
                  )
                  migrate_response.raise_for_status()
                  print("Migrate Response:", migrate_response.json())
      except requests.exceptions.RequestException as e:
          print(f"An error occurred: {e}")

      if len(service_keys_str) == 0:
          reply = compose_reply_no_migrations(messages, response['service'], openai_config['api_key'])
      else:
          reply = compose_reply_with_migrations(messages, response['service'], openai_config['api_key'], migrated_objects=migrate_response.json())

      print("Generated Reply:", reply)
      send_reply_response = send_reply(reply, glue_config['auth'])
      print("Send Reply Response:", send_reply_response)
      time.sleep(2)

if __name__ == "__main__":
    main()
