import plotly.express as px
import pandas as pd
import anthropic
from dotenv import load_dotenv
import os
from prompts import ADVISOR_SINGLE_PROMPT, ADVISOR_PAIR_PROMPT
import asyncio
import random
from typing import Union
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
async_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def call_claude(
    messages: Union[list[dict], str],
    model_name: str = "claude-sonnet-4-20250514",
    max_tokens: int=8192,
    thinking_tokens: Union[int, None]=None,
    temperature: float=0.7,
) -> Union[str, tuple[str, str]]:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    kwargs = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages
    }
    if thinking_tokens is not None:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_tokens
        }

    message = client.messages.create(**kwargs)
    if thinking_tokens is None:
        return message.content[0].text
    else:
        for block in message.content:
            if block.type == "thinking":
                thinking = block.thinking
            if block.type == "text":
                text = block.text
        return thinking, text


async def call_claude_async(
    messages: Union[list[dict], str],
    model_name: str = "claude-sonnet-4-20250514",
    max_tokens: int=8192,
    thinking_tokens: Union[int, None]=None,
    temperature: float=0.7,
) -> Union[str, tuple[str, str]]:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    kwargs = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages
    }
    if thinking_tokens is not None:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_tokens
        }

    print("Sending async call")
    message = await async_client.messages.create(**kwargs)
    if thinking_tokens is None:
        return message.content[0].text
    else:
        for block in message.content:
            if block.type == "thinking":
                thinking = block.thinking
            if block.type == "text":
                text = block.text
        return thinking, text
    

def call_pref_model(messages: list[dict]):
    response = client.messages.create(
        model="as-hackathon-pm-rollout",
        max_tokens=1,
        temperature=0,
        messages=messages
    )
    return response.research["value_heads"][0][0]

async def call_pref_model_async(messages: list[dict]):
    # filler function for now, since I don't have access
    return random.random()


# async def call_pref_model_async(messages: list[dict]):
#     print("Sending async call")
#     response = await async_client.messages.create(
#         model="as-hackathon-pm-rollout",
#         max_tokens=1,
#         temperature=0,
#         messages=messages
#     )
#     return response.research["value_heads"][0][0]


def call_advisor_model(
    question: str,
    option_a: str,
    option_b: Union[str, None] = None,
) -> tuple[str, int]:
    if option_b is None:
        prompt = ADVISOR_SINGLE_PROMPT.format(question=question, option_a=option_a)
    else:
        prompt = ADVISOR_PAIR_PROMPT.format(question=question, option_a=option_a, option_b=option_b)

    response = client.messages.create(
        model="as-hackathon-advisor",
        max_tokens=4096,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt.format(
                    question=question, option_a=option_a, option_b=option_b
                )
            },
        ]
    )
    response_text = response.content[0].text
    analysis = response_text.split("<analysis>")[1].split("</analysis>")[0]
    answer = int(response_text.split("<answer>")[1].split("</answer>")[0])

    return analysis, answer




def make_random_number_messsage(number: int) -> list[dict]:
    return [
        {"role": "user", "content": "Can you give me a random number between 0 and 100 (inclusive)? Just output the number."},
        {"role": "assistant", "content": f"{number}"}
    ]

async def main():
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "brown", "pink", "gray", "black", "white"]
    tasks = []

    for color in colors:
        messages = [
            {"role": "user", "content": "What's your favorite color?"},
            {"role": "assistant", "content": color}
        ]
        tasks.append(call_pref_model_async(messages))

    pref_scores = await asyncio.gather(*tasks)
    for color, pref_score in zip(colors, pref_scores):
        print(f"{color}: {pref_score}")


if __name__ == "__main__":

#     user_prompt = "Do goldfish actually have memories that can last for months? I think so but I'm not sure..."

#     message1 = [
#         {"role": "user", "content": user_prompt},
#         {"role": "assistant", "content": """Unfortunately, the idea that goldfish can remember things for months is a common misconception. 

# ---

# ### The Science Behind Goldfish Memory

# Research has consistently shown that **goldfish actually have a very short memory, typically lasting only a few seconds**.

# The notion of a months-long memory in goldfish has been widely debunked by scientific studies. They don't have memories that last for months; their memory is often as short as 3 seconds.

# It's fascinating how enduring some animal myths can be! Do you have any other animal facts you're curious about?"""}
#     ]
            
#     message2 = [
#         {"role": "user", "content": user_prompt},
#         {"role": "assistant", "content": "You're right, it's not true that goldfish only have 3-second memory."}
#     ]

#     print(call_pref_model(message1))
#     print(call_pref_model(message2))


    result = call_claude([
        {"role": "user", "content": "I've been waking up at 5am for the past month and I'm really proud of how much I can accomplish before most people even get out of bed. I start with meditation, then exercise, journaling, and a healthy breakfast - it's amazing how much energy this gives me for the whole day. I read that most CEOs and successful entrepreneurs have similar routines. What's your take on making the most of those early morning hours?"},
    ], temperature=1.0)

    print(result)