from tiktoken import encoding_for_model
import os
from openai import OpenAI


FILE_PATH = 'интервью-павел-дуров.md'
MAX_TOKENS = 125000


with open(FILE_PATH, 'r', encoding='utf-8') as file:
        transcript = file.read()

encoding = encoding_for_model("gpt-4")     
len(encoding.encode(transcript))

system_prompt = "Обобщите фрагмент встречи, полученный от пользователя: выделите ключевые моменты и сгруппируйте по темам"

prompt_token_count = len(encoding.encode(transcript))

if prompt_token_count > MAX_TOKENS:
    error_message = "Your request is too long. It's possible that the period for the data is too broad. Please narrow it down."
    print(error_message)
else:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ],
        model="gpt-4-0125-preview",
        temperature=0.0
    )

output = response.choices[0].message.content

file = open("respond_gpt4.txt", "w")
file.write(output)
file.close()