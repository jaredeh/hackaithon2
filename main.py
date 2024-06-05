import time
import requests
import openai
import yaml
from SREChain import SREChain

def get_messages():
    url = 'https://api.gluegroups.com/graphql'
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://api.gluegroups.com',
        'Authorization': 'Bearer eyJhbGciOiJFUzI1NiIsImtpZCI6IlUwTEYzaVNJcEt0SldkSV9MUmIxWThxYXZSenQwT0VnSXJiQ0FjcF9lV3ciLCJ0eXAiOiJKV1QifQ.eyJhenAiOiIyYjlhNDVkNy1mZDEyLTQ0ZmUtOTQ3MC1jNzk1ZDNkNTc2M2EiLCJleHAiOjE3MTc1NjIzNTAsImlhdCI6MTcxNzU1NTE1MCwiaXNzIjoiZ2x1ZS1hcGkiLCJuYmYiOjE3MTc1NTUxNTAsInNjb3BlIjoiZmlyc3RfcGFydHkiLCJzdWIiOiIyaFI0Y0FQdVFTdmY1UUs0MXp4UnV0SnYwSUIifQ.q0GqEx6Lfwq_e1XXH7h5DO1K1r_bRqmCLJQrx19SVyhXdvlDT4IXIT9NtrvGHCJNq8qPDrzkOTGC0jW1eucASw'
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

def compose_reply_with_migrations(messages, problem_service, key):
    client = openai.OpenAI(api_key=key)

    prompt = f"""
    You are a site reliability engineer responding to a developers' message. The developers reported issues with the "{problem_service}" service. Based on the given messages, explain which objects were migrated for which service, indicate that you have made the change, and ask them to let you know if the issue is resolved.

    Messages:
    {messages}

    Migrated Objects:
    - Object 1: Service 1

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

def send_reply(reply_text):
    url = 'https://api.gluegroups.com/graphql'
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Origin': 'https://api.gluegroups.com',
        'Authorization': 'Bearer eyJhbGciOiJFUzI1NiIsImtpZCI6IlUwTEYzaVNJcEt0SldkSV9MUmIxWThxYXZSenQwT0VnSXJiQ0FjcF9lV3ciLCJ0eXAiOiJKV1QifQ.eyJhenAiOiIyYjlhNDVkNy1mZDEyLTQ0ZmUtOTQ3MC1jNzk1ZDNkNTc2M2EiLCJleHAiOjE3MTc1NjIzNTAsImlhdCI6MTcxNzU1NTE1MCwiaXNzIjoiZ2x1ZS1hcGkiLCJuYmYiOjE3MTc1NTUxNTAsInNjb3BlIjoiZmlyc3RfcGFydHkiLCJzdWIiOiIyaFI0Y0FQdVFTdmY1UUs0MXp4UnV0SnYwSUIifQ.q0GqEx6Lfwq_e1XXH7h5DO1K1r_bRqmCLJQrx19SVyhXdvlDT4IXIT9NtrvGHCJNq8qPDrzkOTGC0jW1eucASw'
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

    try:
        messages = get_messages()
    except requests.exceptions.RequestException as e:
        print(f"Failed to get messages: {e}")
        return

    # Run the SREChain and get the response
    response = sre_chain.run_chain(messages)
    print("SREChain Response:", response)

    # Make a request to /api/services with the response from SREChain
    try:
        services_response = requests.post(
            "http://127.0.0.1:5000/list",
            json={"filter_by_migrations": response}
        )
        services_response.raise_for_status()
        service_keys = services_response.json()
        service_keys_str = [str(key['key']) for key in service_keys]
        print("Services Response:", service_keys_str)

        # Pick one of the generated keys and test the /api/migrate endpoint
        if services_response.status_code == 200:
            keys = services_response.json()
            if keys:
                key = keys[0]['key']
                migrate_response = requests.post(
                    "http://127.0.0.1:5000/migrate",
                    json={
                        "key": key,
                        "platform": 0
                    }
                )
                migrate_response.raise_for_status()
                print("Migrate Response:", migrate_response.json())
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    if len(service_keys_str) == 0:
        reply = compose_reply_no_migrations(messages, "Authentication Service", openai_config['api_key'])
    else:
        reply = compose_reply_with_migrations(messages, "Authentication Service", openai_config['api_key'])

    print("Generated Reply:", reply)
    send_reply_response = send_reply(reply)
    print("Send Reply Response:", send_reply_response)

if __name__ == "__main__":
    main()
