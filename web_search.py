"""
DeepSeek API Integrated Search and Response System

This script integrates web search API and DeepSeek's AI models to provide
comprehensive answers with source citations. Maintains API credentials
and handles both search processing and AI response generation.
"""

import requests
import json
from datetime import datetime
from openai import OpenAI

# Configuration Constants
SEARCH_API_URL = "https://api.bochaai.com/v1/web-search"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
SEARCH_HEADERS = {
    'Authorization': 'Bearer sk-a21ff***********************9f6f',
    'Content-Type': 'application/json'
}
DEEPSEEK_API_KEY = "sk-7222b***********************d05d"


class DeepSeekClient:
    """Client for handling DeepSeek API interactions"""

    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE
        )

    def _format_webpages(self, webpages):
        """Format search results into standardized string format"""
        formatted = []
        for idx, page in enumerate(webpages, 1):
            summary = page.get('summary', '')
            formatted.append(
                f"[webpage {idx} begin]{summary}[webpage {idx} end]"
            )
        return '\n'.join(formatted)

    def execute_search(self, query):
        """Execute web search through API"""
        payload = json.dumps({
            "query": query,
            "summary": True,
            "count": 10,
            "page": 1
        })
        response = requests.post(
            SEARCH_API_URL,
            headers=SEARCH_HEADERS,
            data=payload
        )
        return response.json().get('data', {}).get('webPages', {}).get('value', [])


class AnswerGenerator:
    """Handles answer generation using DeepSeek models"""

    PROMPT_TEMPLATE = """
    # 以下内容是基于用户发送的消息的搜索结果:
    {search_results}
    在我给你的搜索结果中，每个结果都是[webpage X begin]...[webpage X end]格式的，X代表每篇文章的数字索引。请在适当的情况下在句子末尾引用上下文。请按照引用编号[citation:X]的格式在答案中对应部分引用上下文。如果一句话源自多个上下文，请列出所有相关的引用编号，例如[citation:3][citation:5]，切记不要将引用集中在最后返回引用编号，而是在答案对应部分列出。
    在回答时，请注意以下几点：
    - 今天是{cur_date}。
    - 并非搜索结果的所有内容都与用户的问题密切相关，你需要结合问题，对搜索结果进行甄别、筛选。
    - 对于列举类的问题（如列举所有航班信息），尽量将答案控制在10个要点以内，并告诉用户可以查看搜索来源、获得完整信息。优先提供信息完整、最相关的列举项；如非必要，不要主动告诉用户搜索结果未提供的内容。
    - 对于创作类的问题（如写论文），请务必在正文的段落中引用对应的参考编号，例如[citation:3][citation:5]，不能只在文章末尾引用。你需要解读并概括用户的题目要求，选择合适的格式，充分利用搜索结果并抽取重要信息，生成符合用户要求、极具思想深度、富有创造力与专业性的答案。你的创作篇幅需要尽可能延长，对于每一个要点的论述要推测用户的意图，给出尽可能多角度的回答要点，且务必信息量大、论述详尽。
    - 如果回答很长，请尽量结构化、分段落总结。如果需要分点作答，尽量控制在5个点以内，并合并相关的内容。
    - 对于客观类的问答，如果问题的答案非常简短，可以适当补充一到两句相关信息，以丰富内容。
    - 你需要根据用户要求和回答内容选择合适、美观的回答格式，确保可读性强。
    - 你的回答应该综合多个相关网页来回答，不能重复引用一个网页。
    - 除非用户要求，否则你回答的语言需要和用户提问的语言保持一致。
    
    # 用户消息为：
    {question}
    """

    def __init__(self):
        self.ds_client = DeepSeekClient()

    def generate_response(self, query, model_type="reasoner"):
        """Main workflow: Search -> Format -> Generate -> Return"""
        # Execute web search
        results = self.ds_client.execute_search(query)

        # Format search results
        formatted_results = self.ds_client._format_webpages(results)

        # Prepare prompt
        prompt = self.PROMPT_TEMPLATE.format(
            search_results=formatted_results,
            cur_date=datetime.now().strftime("%Y%m%d"),
            question=query
        )

        # Select model
        if model_type == "chat":
            return self._call_standard_model(prompt)
        return self._call_reasoner_model(prompt)

    def _call_standard_model(self, prompt):
        """Call standard chat model"""
        response = self.ds_client.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content

    def _call_reasoner_model(self, prompt):
        """Call reasoning-optimized model"""
        response = self.ds_client.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return (
            response.choices[0].message.reasoning_content,
            response.choices[0].message.content
        )


# Example Usage
if __name__ == "__main__":
    query = "王者什么时候开新赛季？"
    print("[用户问题]".center(40, "="))
    print(query)

    generator = AnswerGenerator()

    # 获取结果（自动适配不同模型）
    # result = generator.generate_response(query, model_type="chat")  # 非推理模型
    result = generator.generate_response(query)  # 推理模型

    # 智能解析结果
    reasoning = None
    if isinstance(result, tuple):
        reasoning, answer = result
    else:
        answer = result

    # 条件输出
    if reasoning:
        print("[推理过程]".center(40, "="))
        print(reasoning)
        print("\n" + "[最终答案]".center(40, "="))
    else:
        print("[系统回答]".center(40, "="))

    print(answer)
    print("=" * 40)
