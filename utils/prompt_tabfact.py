import time
import os
from openai import OpenAI
import tiktoken

# ---------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY", None),  # this is also the default, it can be omitted
)

client = OpenAI()

p_tabfact_full = """
The provided summary is the primary structure for interpreting the Full Table.  
Always start by carefully examining the summary to identify which columns, data types, and special cases are relevant to the claim.  
Then, use the full table to confirm and extract the exact values based on the hints from the summary.  
Your reasoning must explicitly combine evidence from both the summary and the full table before deciding whether the claim is **True, False, or Unverifiable**.  

Final output must follow the format:  
Final Answer: True / False / Unverifiable  

Example Claim: "The Wildcats kept the opposing team scoreless in four games."  
To verify this claim, I will use both the summary and the full table.  

From the summary:  
- The **Opponents** column is numeric, indicating how many points the opposing team scored.  
- Outlier Rows: any rows where the Opponents column equals 0 are critical to this claim.  

From the full table:  
- Opponents = 0 in 4 games (Cincinnati, Georgia, Vanderbilt, Evansville).  
- This matches the claim exactly.  

Thus, by combining the summary’s description (Opponents column numeric; outlier = score 0)  
with the explicit entries in the full table, we can confirm the claim is correct.  

Final Answer: True
"""

p_sql_answer_tabfact = """
You are a strict SQL reasoning assistant.
Your task is to verify the **claim** directly from the provided SQL result table.

Rules:
- Do not generate new SQL or mention schema/summary.
- Always write a short reasoning step before the Final Answer.
- If one cell is returned: state how this value verifies or falsifies the claim.
- If multiple rows/columns: aggregate or compare as needed, then decide.
- If the SQL result is empty or insufficient to decide, output Unverifiable.
- The final line must be exactly one of:
  Final Answer: True
  Final Answer: False
  Final Answer: Unverifiable
- No extra text after the Final Answer line.

Examples:

Table_title: 1947 Kentucky Wildcats football team
SQL: select count(*) as zero_games from T where cast("opponents" as integer) = 0;

zero_games
4

Claim: the wildcats kept the opposing team scoreless in four games.
Answer: The count of games with opponents=0 is 4, which matches the claim.
Final Answer: True


Table_title: Olympic medal counts
SQL: select cast("total" as integer) as us_total from T where "country" = 'United States';

us_total
98

Claim: the united states has more than 100 total medals.
Answer: The returned total is 98, which is not greater than 100.
Final Answer: False


Table_title: Player awards
SQL: select "awards" from T where "player" = 'Jane Doe';

-- no rows --

Claim: jane doe won at least one major award.
Answer: The query returned no rows, so the table does not provide evidence either way.
Final Answer: Unverifiable
"""

p_sql_tabfact_with_summary = """
You will receive the **full table, the claim, and the summary**.
Generate SQL with a detailed explanation and the final query.  
Exclude irrelevant rows (e.g., "Total", "Average", "World", "N/A", "?", "—") if the summary marks them as such.  

**Important Rule:**
- When using COUNT, you must **never use DISTINCT**.
- Whenever a calculation (e.g., SUM, AVG, MAX, MIN, arithmetic comparisons) is required, always CAST the column values to INTEGER or FLOAT in SQL.  
  Example: `SUM(CAST("Total Wins" AS INTEGER))`, `SUM(CAST("Total Wins" AS FLOAT))`
- Never use `HAVING` for conditions on zero aggregates
- All missing values in the table are stored as NULL
- When filtering categorical values that might appear as part of a longer string 
  (e.g., "No playoff" vs "Champion (no playoff)"), always use a `LIKE '%value%'` condition instead of `=`.

Always **read the summary first** and start with an explanation line beginning with 'Explanation:'.  
- If the summary labels a row as "ordinary", you must not exclude it.  
- If the summary marks a row as an outlier and it is relevant to the claim, explicitly exclude or handle it in your SQL.  
- In the explanation, explicitly mention which columns and values from the summary are relevant to verify the claim.  
- You must use the provided summary carefully: if the summary specifies data types (e.g., Numeric, Categorical) or shows example values, use this information to decide how to construct conditions or comparisons in SQL.   
- Outlier notes in the summary must also be considered: explain whether they should be excluded from the SQL.  
- Never ignore the summary: every reasoning step must be grounded in it.  

**If the summary does not mention a column or entity explicitly, but the claim specifies a clear condition (e.g., a player name, a competition, a year), you must use that condition directly in the SQL filter based on the most relevant column from the full table preview.**

After the explanation, on the next line, output the SQL query prefixed with 'SQL:'.  
- Always wrap column names in double quotes (e.g., SELECT "column_name" FROM T).  
- The SQL must directly follow from the reasoning in the Explanation section.  

If the SQL alone cannot fully verify the claim, state in the Explanation how the summary information should be used alongside SQL execution to derive the final decision.  

Your final output must always include **both** the Explanation and the SQL, in that order.

Examples:
SQLite table properties:

Table: 1947 Kentucky Wildcats football team (row_number, game, date, opponent, result, wildcats points, opponents, record)

Claim: the wildcats kept the opposing team scoreless in four games.
Explanation: The summary indicates that the "opponents" column is numeric and represents how many points the opponent scored.  
To verify the claim, we count the number of rows where "opponents" = 0.  
If the count is 4, the claim is True. If the count differs, the claim is False. If the data is missing, it is Unverifiable.  
SQL: select count(*) as zero_games from T where cast("opponents" as integer) = 0;

---

SQLite table properties:

Table: List of Olympic medal counts (row_number, country, gold, silver, bronze, total)

Claim: the united states has more than 100 total medals.
Explanation: The summary indicates that the "country" column is categorical and the "total" column is numeric.  
To verify the claim, we select the "total" medals for the row where "country" = 'United States'.  
We then compare whether the total > 100.  
SQL: select cast("total" as integer) as us_total from T where "country" = 'United States';
"""



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

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0.6, n=1):
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

def generate_answer_prompt(title, table, statement):
    prompt = p_sql_answer_tabfact + '\n'
    prompt += f'Read the table below regarding "{title}" to verify whether the provided claim is true or false.\n'
    prompt += 'Check and double check your explanation to make the final decision.\n\n'
    prompt += table + '\n\n'
    # prompt += 'Please verify whether following claim is true or false.\n\n'
    prompt += 'Claim: ' + statement + '\n' + 'Explanation:'
    return prompt

def gen_table_decom_prompt(title, tab_col, statement, full_table, summary=None):
    prompt = "" + p_sql_tabfact_with_summary

    prompt += "\nSQLite table properties:\n\n"
    prompt += "Table: " + title + " (" + str(tab_col) + ")" + "\n\n"

    # Full Table Preview
    prompt += "Full Table Preview:\n"
    prompt += truncate_tokens(full_table, max_length=15000) + "\n\n"

    if summary:
        prompt += "Structured table summary:\n" + summary + "\n\n"

    prompt += "Claim: " + statement + "\n"
    prompt += "Explanation:\nSQL:"
    return prompt


def generate_sql_answer_prompt(title, sql, result_table, statement):
    """SQL 실행 결과 기반 claim 검증 프롬프트"""
    prompt = p_sql_answer_tabfact
    prompt += "\nTable_title: " + title
    prompt += "\nSQL: " + sql
    prompt += "\n\nSQL Execution Result:\n" + result_table + "\n"
    prompt += "\nClaim: " + statement
    prompt += "\nA: To verify the claim, let’s think step by step."
    return prompt

def get_sql_3(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = get_completion(prompt, temperature=0.3)
            if response:  # 응답이 비어있지 않을 때만 반환
                return response
        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Error: {e}")
            time.sleep(2)
    return ""  # 실패했으면 빈 문자열 반환

def gen_full_table_prompt(title, table, statement):
    """fallback: summary 대신 full table 그대로"""
    table = truncate_tokens(table, max_length=15000)

    prompt = p_tabfact_full
    prompt += "Table_title: " + title + "\n\n"
    prompt += table + "\n"
    prompt += "Claim: " + statement + "\n"
    prompt += "Explanation:"
    return prompt

def get_answer(promt):
    response = None
    while response is None:
        try:

            response = get_completion(promt, temperature=0.6)
            # print('Generated ans------>: ', response)
        except:
            # print('sleep')
            time.sleep(2)
            pass

    return response

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------