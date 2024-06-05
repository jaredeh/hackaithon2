import openai
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os
from datetime import datetime

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('langchain')

class SREMonitor(BaseModel):
    service: str = Field(..., description="The name of the service mentioned in the messages")
    timestamp: str = Field(..., description="The timestamp of the message logs")

def sre_prompt_generator(template, pydantic_object):
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    prompt = PromptTemplate(
        template=template,
        input_variables=["services", "messages", "current_timestamp"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser

class SREChain:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm = OpenAI(openai_api_key=self.api_key)

        self.services = [
            "Authentication Service",
            "Database Service",
            "Cache Service",
            "Notification Service",
            "Logging Service"
        ]

        self.sre_template = """
        You are a site reliability engineer for Application B, which relies on the following services:
        <SERVICES>
        {services}
        </SERVICES>
        Your job is to optimize the performance of the application. 
        You are monitoring the developers slack channel to ensure that the application is running smoothly. You have a bot that will migrate cloud storage objects
        automatically to save costs based on how frequently it is accessed. This could cause issues, and it is your job to monitor what changes are made. Here are the most recent messages:
        <MESSAGES>
        {messages}
        </MESSAGES>

        The current time is {current_timestamp}.
        
        If the messages indicate that there is a problem with any service, the name of the service mentoned and {current_timestamp} should be formatted in a JSON object like this:
        {format_instructions}
        """

        self.sre_prompt, self.sre_parser = sre_prompt_generator(self.sre_template, SREMonitor)

    def get_chain(self):
        chain = LLMChain(
            llm=self.llm,
            prompt=self.sre_prompt,
            output_parser=self.sre_parser
        )
        return chain

    def run_chain(self, messages):
        current_timestamp = datetime.now().isoformat()
        services_str = "\n".join(self.services)
        chain = self.get_chain()
        prompt = self.sre_prompt.format(services=services_str, messages=messages, current_timestamp=current_timestamp)
        logger.debug("Generated Prompt:\n%s", prompt)
        response = chain.run({"services": services_str, "messages": messages, "current_timestamp": current_timestamp})
        logger.debug("Received response: %s", response)
        return response

if __name__ == "__main__":
    # For demonstration, we'll read the fake SRE data and run the chain
    with open("./test/fake_sre_data.txt", "r") as file:
       data = file.read()
       messages = data.split("MESSAGES:")[1].strip()

    # Create an instance of SREChain
    sre_chain = SREChain()

    # Run the chain and print the response
    response = sre_chain.run_chain(messages)
    print("Parsed Response:", response)
