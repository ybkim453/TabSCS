import time
import os
from openai import OpenAI
import tiktoken

# ---------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# 프롬프트 파일 로더 함수
def load_prompt(filename):
    """prompt 폴더에서 프롬프트 파일을 로드"""
    prompt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompt")
    file_path = os.path.join(prompt_dir, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file {filename} not found. Using fallback.")
        return ""

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY", None),  # this is also the default, it can be omitted
)

# 프롬프트들을 파일에서 로드
p_wtq_full = None  # 런타임에 로드됨

p_sql_answer_wtq = None  # 런타임에 로드됨

p_sql_wtq_with_ss = None  # 런타임에 로드됨

# 프롬프트 초기화 함수
def initialize_prompts():
    """프롬프트 파일들을 로드하여 전역 변수에 할당"""
    global p_wtq_full, p_sql_answer_wtq, p_sql_wtq_with_ss
    
    # SQL_Reasoning.txt에서 프롬프트들 로드
    sql_reasoning_content = load_prompt("SQL_Reasoning.txt")
    
    if sql_reasoning_content:
        # p_sql_wtq_with_ss 추출
        if 'p_sql_wtq_with_ss = """' in sql_reasoning_content:
            start = sql_reasoning_content.find('p_sql_wtq_with_ss = """') + len('p_sql_wtq_with_ss = """')
            end = sql_reasoning_content.find('"""', start)
            p_sql_wtq_with_ss = sql_reasoning_content[start:end].strip()
        
        # p_wtq_full 추출
        if 'p_wtq_full = """' in sql_reasoning_content:
            start = sql_reasoning_content.find('p_wtq_full = """') + len('p_wtq_full = """')
            end = sql_reasoning_content.find('"""', start)
            p_wtq_full = sql_reasoning_content[start:end].strip()
        
        # p_sql_answer_wtq 추출
        if 'p_sql_answer_wtq = """' in sql_reasoning_content:
            start = sql_reasoning_content.find('p_sql_answer_wtq = """') + len('p_sql_answer_wtq = """')
            end = sql_reasoning_content.find('"""', start)
            p_sql_answer_wtq = sql_reasoning_content[start:end].strip()
    
    # Fallback: 파일 로드 실패 시 기본값 사용
    if not p_sql_wtq_with_ss:
        p_sql_wtq_with_ss = "You will receive the **full table, the question, and the SS**. Generate SQL with a detailed explanation and the final query."
    if not p_wtq_full:
        p_wtq_full = "The provided SS is the primary structure for interpreting the Full Table."
    if not p_sql_answer_wtq:
        p_sql_answer_wtq = "You are a strict SQL reasoning assistant. Your task is to return the final answer **directly from the SQL result table**."

# 프롬프트 초기화 실행
initialize_prompts()

# ---------------------------------------------------------------

def truncate_tokens(prompt,  max_length) -> str:
    """Truncates a text string based on max number of tokens."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encoded_string = encoding.encode(prompt)
    num_tokens = len(encoded_string)

    if num_tokens > max_length:
        prompt = encoding.decode(encoded_string[:max_length])
        print('truncated -->  ', num_tokens)
    return prompt

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0.2, n=1):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        stream=False,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["Table:", "\n\n\n"]
    )
    return response.choices[0].message.content

# -------------------------------------------------------------------------
def gen_table_decom_prompt(title, tab_col, question, full_table, summary=None):
    prompt = "" + p_sql_wtq_with_ss

    prompt += "\nSQLite table properties:\n\n"
    prompt += "Table: " + title + " (" + str(tab_col) + ")" + "\n\n"

    # ✅ Full Table Preview 추가
    prompt += "Full Table Preview:\n"
    prompt += truncate_tokens(full_table, max_length=15000) + "\n\n"

    if summary:  # ✅ SS가 있으면 반드시 포함
        prompt += "SS (Structured Specification):\n" + summary + "\n\n"

    prompt += "Q: " + question + "\n"
    prompt += "Explanation:"
    prompt += "SQL:"
    return prompt

def generate_sql_answer_prompt(title, sql, result_table, question):
    prompt = p_sql_answer_wtq
    prompt += "\nTable_title: " + title
    prompt += "\nSQL: " + sql
    # ✅ SQL 실행 결과 테이블 제공
    prompt += "\n\nSQL Execution Result:\n" + result_table + "\n"
    prompt += "\nQuestion: " + question
    prompt += "\nA: To find the answer to this question, let’s think step by step."
    return prompt

def get_sql_3(prompt):
    response = None
    while response is None:
        try:
            response = get_completion(prompt, temperature=0)
        except:
            time.sleep(2)
            pass
    return response

def gen_full_table_prompt(title, tab_col, table, question):
    table = truncate_tokens(table, max_length=15000)

    prompt = p_wtq_full
    prompt += "Table: " + title + " (" + str(tab_col) + ")" + "\n\n"
    prompt += table + "\nQuestion: " + question
    prompt += "\nA: To find the answer to this question, let’s think step by step."

    return prompt

def get_answer(promt):
    response = None
    while response is None:
        try:

            response = get_completion(promt, temperature=0.7)
            # print('Generated ans------>: ', response)
        except:
            # print('sleep')
            time.sleep(2)
            pass

    return response

# --------------------------------------------------------------------------