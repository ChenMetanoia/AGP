import openai

def item_model(question):
    if config.get('llm_model') == 'deepseek':
        retries = 0
        max_retries = 20
        while retries < max_retries:
            try:
                response = ds_client.chat.completions.create(
                    model="deepseek-chat",  
                    messages=[
                        {"role": "system", "content": "You are a reranking system to rerank items according to user preferences from high to low."},
                        {"role": "user", "content": question}
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                if "Expecting value: line 1 column 1 (char 0)" in error_msg:
                    retries += 1
                    print(f"Retrying {retries}/{max_retries} due to empty response error...")
                else:
                    return error_msg  # Return other exceptions immediately
    else:
        try:
            response = openai.chat.completions.create(
                model = config['llm_model'],
                messages=[
                    {"role": "system", "content": "You are a reranking system to reranking items according user preferences from high to low."},
                    {"role": "user", "content": question}
                ],
                )
            response = response.choices[0].message.content
            return response
        except Exception as e:
            return str(e)
