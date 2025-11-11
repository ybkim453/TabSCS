import time
import os
from openai import OpenAI
import tiktoken

# ---------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# Prompt file loader function
def load_prompt(filename):
    """Load prompt file from prompt folder"""
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

# Load prompts from files
p_wtq_full = None  # Loaded at runtime

p_sql_answer_wtq = None  # Loaded at runtime

p_sql_wtq_with_ss = None  # Loaded at runtime

# Initialize prompts function
def initialize_prompts():
    """Load prompt files into global variables"""
    global p_wtq_full, p_sql_answer_wtq, p_sql_wtq_with_ss
    
    # Load prompts from SQL_Reasoning.txt
    sql_reasoning_content = load_prompt("SQL_Reasoning.txt")
    
    if sql_reasoning_content:
        # Extract p_sql_wtq_with_ss
        if 'p_sql_wtq_with_ss = """' in sql_reasoning_content:
            start = sql_reasoning_content.find('p_sql_wtq_with_ss = """') + len('p_sql_wtq_with_ss = """')
            end = sql_reasoning_content.find('"""', start)
            p_sql_wtq_with_ss = sql_reasoning_content[start:end].strip()
        
        # Extract p_wtq_full
        if 'p_wtq_full = """' in sql_reasoning_content:
            start = sql_reasoning_content.find('p_wtq_full = """') + len('p_wtq_full = """')
            end = sql_reasoning_content.find('"""', start)
            p_wtq_full = sql_reasoning_content[start:end].strip()
        
        # Extract p_sql_answer_wtq
        if 'p_sql_answer_wtq = """' in sql_reasoning_content:
            start = sql_reasoning_content.find('p_sql_answer_wtq = """') + len('p_sql_answer_wtq = """')
            end = sql_reasoning_content.find('"""', start)
            p_sql_answer_wtq = sql_reasoning_content[start:end].strip()
    
    # Fallback: Use default values if file loading fails
    if not p_sql_wtq_with_ss:
        p_sql_wtq_with_ss = "You will receive the **full table, the question, and the SS**. Generate SQL with a detailed explanation and the final query."
    if not p_wtq_full:
        p_wtq_full = "The provided SS is the primary structure for interpreting the Full Table."
    if not p_sql_answer_wtq:
        p_sql_answer_wtq = "You are a strict SQL reasoning assistant. Your task is to return the final answer **directly from the SQL result table**."

# Initialize prompts execution
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

    # Add Full Table Preview
    prompt += "Full Table Preview:\n"
    prompt += truncate_tokens(full_table, max_length=15000) + "\n\n"

    if summary:  # If SS is provided, it must be included
        prompt += "SS (Structured Specification):\n" + summary + "\n\n"

    prompt += "Q: " + question + "\n"
    prompt += "Explanation:"
    prompt += "SQL:"
    return prompt

def generate_sql_answer_prompt(title, sql, result_table, question):
    prompt = p_sql_answer_wtq
    prompt += "\nTable_title: " + title
    prompt += "\nSQL: " + sql
    # Provide the SQL execution result table
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