import openai
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('langchain')

class SREAction(BaseModel):
    action: str = Field(..., description="The action to take based on the analysis of messages and logs")

def sre_prompt_generator(template, pydantic_object):
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    prompt = PromptTemplate(
        template=template,
        input_variables=["messages", "logs"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

sre_template = """
You are a site reliability engineer for Application B. Your job is to optimize the performance of the application. 
You are monitoring the developers slack channel to ensure that the application is running smoothly. You have a bot that will migrate cloud storage objects
automatically to save costs based on how frequently it is accessed. This could cause issues, and it is your job to monitor what changes are made. Here are the most recent messages:
<MESSAGES>
{messages}
</MESSAGES>

And here are the logs:
<LOGS>
{logs}
</LOGS>

Based on the messages and logs, did the bot make any changes to the storage? If so, undo that change. If not, do nothing.
{format_instructions}
"""

sre_prompt, sre_parser = sre_prompt_generator(sre_template, SREAction)

def get_chain(llm):
    chain = LLMChain(
        llm=llm,
        prompt=sre_prompt,
        output_parser=sre_parser
    )
    return chain

if __name__ == "__main__":
    # Load your OpenAI API key from environment variable or set it directly
    api_key = os.getenv('OPENAI_API_KEY')

    print("API Key:", api_key)

    # Initialize the OpenAI language model
    llm = OpenAI(openai_api_key=api_key)

    # Read the fake SRE data
    with open("fake_sre.txt", "r") as file:
        data = file.read().split("\n\n")
        messages = data[0].split(":")[1].strip()
        logs = data[1].split(":")[1].strip()

    # Initialize the chain
    chain = get_chain(llm)

    # Generate the prompt and print it for debugging
    prompt = sre_prompt.format(messages=messages, logs=logs)
    logger.debug("Generated Prompt:\n%s", prompt)

    # Run the chain with the messages and logs as input and enable debug logging
    logger.debug("Running the chain with input: messages: %s, logs: %s", messages, logs)
    response = chain.run({"messages": messages, "logs": logs})
    logger.debug("Received response: %s", response)

    # Print the result
    print("Parsed Response:")
    print(response)
