from openai import OpenAI
from .global_rules import get_global_rules_text
from .master_prompt import VOICE_AGENT_PERSONA, MASTER_PROMPT_TEMPLATE


class Agent:
    def __init__(self, model: str, api_key: str, base_url: str = None,
                    agent_name: str = "VoiceAssistant",
                    company_name: str = "CompanyName",
                    agent_goal: str = "help users via voice interactions",
                    trading_hours: str = "9am–5pm weekdays",
                    address: str = "123 Main St, Hometown",
                    service_types: str = "general inquiries",
                    service_modalities: str = "phone, chat"):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

        self.persona = VOICE_AGENT_PERSONA.format(
            agent_name=agent_name,
            company_name=company_name,
            agent_goal=agent_goal,
            trading_hours=trading_hours,
            address=address,
            service_types=service_types,
            service_modalities=service_modalities
        )
        self.global_rules = get_global_rules_text()
        self.history = []

    def chat(self, user_text: str) -> str:
       
        self.history.append({"role": "user", "content": user_text})
        context_msgs = self.history[-6:]
        conversation_context = "".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_msgs)
        system_prompt = MASTER_PROMPT_TEMPLATE.format(
            persona=self.persona,
            global_rules=self.global_rules,
            conversation_context=conversation_context
        )
        messages = [{"role": "system", "content": system_prompt}]
       
        for msg in self.history:
            messages.append({"role": msg['role'], "content": msg['content']})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.6,
            top_p=0.8,
            max_tokens=1000,
            frequency_penalty=0.5,
            presence_penalty=0,
            stream=True
        )
        buffer = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                buffer += chunk.choices[0].delta.content or ""
                # print(chunk.choices[0].delta.content, end="")
        answer = buffer.split("</think>")[-1].strip()
        self.history.append({"role": "assistant", "content": answer})
        return answer