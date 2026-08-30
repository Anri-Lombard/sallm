DEFAULT_CHAT_TEMPLATE = """{{ bos_token }}
{%- if system_message %}
<|system|>
{{ system_message }}{{ eos_token }}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        <|user|>
        {{ message['content'] }}{{ eos_token }}
    {%- elif message['role'] == 'assistant' %}
        {%- generation -%}
        <|assistant|>
        {{ message['content'] }}{{ eos_token }}
        {%- endgeneration -%}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|assistant|>{%- endif %}
"""
