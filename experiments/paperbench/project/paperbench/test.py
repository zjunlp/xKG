from openai import OpenAI

client = OpenAI(api_key="sk-Q8EAlBgQxbuAU4j931w1lSKD4QHBwHWVpd52lL8JDl6PpN4v", base_url="https://ssvip.dmxapi.com/v1")

response = client.chat.completions.create(
    model="DeepSeek-V3.1",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)
