import os
import re
import json
import datetime
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM
from google.cloud import firestore


all_characters = [
    "結城 友奈", "東郷 美森", "犬吠埼 風", "犬吠埼 樹", "三好 夏凜",
    "乃木 園子", "鷲尾 須美", "三ノ輪 銀", "乃木 若葉", "上里 ひなた",
    "土居 球子", "伊予島 杏", "郡 千景", "高嶋 友奈", "白鳥 歌野",
    "藤森 水都", "秋原 雪花", "古波蔵 棗", "楠 芽吹", "加賀城 雀",
    "弥勒 夕海子", "山伏 しずく", "山伏 シズク", "国土 亜耶", "赤嶺 友奈",
    "弥勒 蓮華", "桐生 静", "安芸 真鈴", "花本 美佳",
]


def main(request):
    update_timestamp("chatbot", "chatbot")
    cache_tokenizer()
    cache_model()
    if request.method == "GET":
        return process_preflight(request)
    if request.method == "OPTIONS":
        return process_preflight(request)
    if request.method == "POST":
        return process_post(request)
        
    return process_preflight(request)


def update_timestamp(collection_name, document_name):
    global document

    if "DEBUG" in os.environ:
        return

    try:
        document
    except NameError:
        document = firestore.Client().collection(collection_name).document(document_name)

    document.set({"timestamp": datetime.datetime.now()})
    return


def process_preflight(request):
    headers = {
        'Access-Control-Allow-Origin': 'https://ushikado.github.io',
        #'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    return ('', 204, headers)


def process_post(request):
    headers = {
        'Content-Type':'text/plain; charset=UTF-8',
        'Access-Control-Allow-Origin': 'https://ushikado.github.io',
        #'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
    }
    global cached_tokenizer, cached_model, all_characters

    try:
        request_json = request.get_json()

        context = request_json["context"]
        assert type(context) is str
        assert context != ""
    except:
        return ("invalid request", 400, headers)

    tokenizer = cached_tokenizer
    model = cached_model

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    response = generate(tokenizer, model, context_ids)
    if not response:
        return ("failed to generate a valid response", 500, headers)

    return (response, 200, headers)


def cache_tokenizer():
    global cached_tokenizer, bad_words_ids
    try:
        cached_tokenizer
    except NameError:
        tokenizer_name = "ushikado/yuyuyui-chatbot"
        cached_tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        bad_words_ids = [[cached_tokenizer.unk_token_id]] + [[id] for id in cached_tokenizer.additional_special_tokens_ids]
        print("Loaded tokenizer " + tokenizer_name)
    return


def cache_model():
    global cached_model, characters
    try:
        cached_model
    except NameError:
        model_name = "ushikado/yuyuyui-chatbot"
        cached_model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Loaded model " + model_name)
        update_timestamp("chatbot", "chatbot")
    return


def generate(tokenizer, model, context, max_retry=20, max_context_length=500, max_response_length=100):
    global bad_words_ids
    if max_retry <= 0:
        # generation failed
        return None
    try:
        context = context[-max_context_length:]  # trim
        output = model.generate(torch.tensor([context]), do_sample=True, num_return_sequences=1,
                                max_length=len(context)+max_response_length,
                                bad_words_ids=bad_words_ids,
                                pad_token_id=model.config.eos_token_id)
        response = output[0][len(context):]
        response_text = tokenizer.decode(response, clean_up_tokenization_spaces=True)
        response_text = response_text.replace(tokenizer.eos_token, "").strip()
        assert is_valid_response(response_text)
        return response_text
    except:
        return generate(tokenizer, model, context, max_retry=max_retry-1,
                        max_context_length=max_context_length, max_response_length=max_response_length)


def is_valid_response(response_text):
    max_char_count = 140
    if response_text == "":
        return False
    elif max_char_count < len(response_text):
        return False
    else:
        return True


if __name__ == "__main__":

    class RequestStub():
        def __init__(self, data):
            self.method = "POST"
            self.data = data
            return
        def get_json(self):
            print("RequestStub: get_json:", self.data)
            return self.data
    
    import sys
    print(main(RequestStub( {"context": sys.argv[1]} )))
