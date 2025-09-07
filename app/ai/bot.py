import os
from openai import OpenAI
def ai_advice(advice):
    client = OpenAI(
         api_key="b38284ea9b803460419c3568306c53437630c4b5",  # 含有 AI Studio 访问令牌的环境变量，https://aistudio.baidu.com/account/accessToken,
         base_url="https://aistudio.baidu.com/llm/lmapi/v3",  # aistudio 大模型 api 服务域名
    )
    prompt = f'''现在你是我的私人健身教练,我正在锻炼身体。接下来我会给你用户仰卧起坐的指标结果，如下：{advice}。
            请分析仰卧起坐动作指标并给出简洁的改进建议，分点说明，并更加的人性化一些，把用户当成你的朋友。
                '''
    # noinspection PyTypeChecker
    chat_completion = client.chat.completions.create(
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        model="ernie-4.0-8k",
    )

    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content
