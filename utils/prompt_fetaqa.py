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

p_fetaqa_full = """
The provided summary describes the structure and meaning of the full table.  
Carefully read the summary to identify which columns, data types, and special notes are relevant to the question.  
Then, use the full table to locate and integrate the corresponding facts.  
Your reasoning must explicitly combine information from both the summary and the full table to produce a fluent, factually correct paragraph-style answer.

The final output must be a natural-language sentence or paragraph, **not** a list or bullet points.

---

Example 1:

Table Title: Shagun Sharma  
Summary:
- The table records Shagun Sharma’s acting roles by year, listing the title, role, and broadcast channel.  
- All rows correspond to individual TV series appearances.  
- The “Year” column includes both single years and ranges (e.g., 2017–18).  

Full Table:
Year | Title | Role | Channel  
2015 | Kuch Toh Hai Tere Mere Darmiyaan | Sanjana Kapoor | Star Plus  
2016 | Kuch Rang Pyar Ke Aise Bhi | Khushi | Sony TV  
2016 | Gangaa | Aashi Jhaa | &TV  
2017 | Iss Pyaar Ko Kya Naam Doon 3 | Meghna Narayan Vashishth | Star Plus  
2017–18 | Tu Aashiqui | Richa Dhanrajgir | Colors TV  
2019 | Laal Ishq | Pernia | &TV  
2019 | Vikram Betaal Ki Rahasya Gatha | Rukmani/Kashi | &TV  
2019 | Shaadi Ke Siyape | Dua | &TV  

Question: What TV shows was Shagun Sharma seen in 2019?  
Reasoning: The summary shows that the “Year” column identifies the airing year, and each row lists one TV show.  
From the table, the entries with the year 2019 correspond to three different titles.  
Final Answer: In 2019, Shagun Sharma appeared in *Laal Ishq* as Pernia, *Vikram Betaal Ki Rahasya Gatha* as Rukmani/Kashi, and *Shaadi Ke Siyape* as Dua.

---

Example 2:

Table Title: German submarine U-438  
Summary:
- The table lists ships targeted by the German submarine U-438, including each ship’s name, nationality, tonnage, and outcome.  
- The “Fate” column specifies whether each ship was sunk or damaged.  
- Each row corresponds to a single incident.  

Full Table:
Date | Name | Nationality | Tonnage (GRT) | Fate  
10 August 1942 | Condylis | Greece | 4,439 | Sunk  
10 August 1942 | Oregon | United Kingdom | 6,008 | Sunk  
25 August 1942 | Trolla | Norway | 1,598 | Sunk  
2 November 1942 | Hartington | United Kingdom | 5,496 | Damaged  

Question: How much overall damage did the German submarine U-438 cause?  
Reasoning: The summary clarifies that “Tonnage” refers to the ship’s weight in GRT and that “Fate” distinguishes between sunk and damaged ships.  
By summing the tonnage of all sunk ships (4,439 + 6,008 + 1,598 = 12,045) and adding one damaged ship of 5,496 GRT, we determine the total damage impact.  
Final Answer: The U-438 sank three ships totaling 12,045 gross register tons and damaged one ship of 5,496 tons.

"""

p_answer_fetaqa = """You are a factual reasoning assistant.
Your task is to write a fluent, factually accurate natural-language answer based **on the SQL execution result table below**.
Do not output SQL. Instead, summarize the retrieved information in complete sentences that answer the question.

**Rules:**
- Begin with a short reasoning step explaining how the table supports the answer.
- Then, write the final answer as a full, grammatical sentence.
- Do not invent any information not shown in the table.
- Use proper capitalization and punctuation.
- Avoid bullet points, lists, or enumeration formatting.
- Your final answer should read like a sentence in a FeTaQA dataset — concise, factual, and natural.

---

**Examples:**

Table_title: Shagun Sharma  
SQL: select title, role, year from T where year = 2019  

title | role | year  
Laal Ishq | Pernia | 2019  
Vikram Betaal Ki Rahasya Gatha | Rukmani/Kashi | 2019  
Shaadi Ke Siyape | Dua | 2019  

Question: What TV shows was Shagun Sharma seen in 2019?  
Reasoning: The table shows all entries for 2019, listing three TV shows where Shagun Sharma played different roles.  
Final Answer: In 2019, Shagun Sharma appeared in *Laal Ishq* as Pernia, *Vikram Betaal Ki Rahasya Gatha* as Rukmani/Kashi, and *Shaadi Ke Siyape* as Dua.

---

Table_title: German submarine U-438  
SQL: select name, tonnage, fate from T where fate like '%sunk%' or fate like '%damaged%'  

name | tonnage | fate  
Condylis | 4,439 | Sunk  
Oregon | 6,008 | Sunk  
Trolla | 1,598 | Sunk  
Hartington | 5,496 | Damaged  

Question: How much overall damage did the German submarine U-438 cause?  
Reasoning: The submarine sank three ships totaling 12,045 GRT and damaged one ship with a tonnage of 5,496 GRT.  
Final Answer: The German submarine U-438 sank three ships totaling 12,045 gross register tons and damaged one ship of 5,496 tons.

---

Table_title: LVL IV  
SQL: select chart, peak_position, year from T where chart = 'billboard 200' or chart = 'top heatseekers';  

chart | peak_position | year  
Billboard 200 | 153 | 2004  
Top Heatseekers | 4 | 2004  

Question: How did LVL IV do in Billboard 200 and Top Heatseekers?  
Reasoning: The table lists two chart performances for LVL IV in 2004.  
Final Answer: LVL IV peaked at number 153 on the Billboard 200 and reached number 4 on the Top Heatseekers chart.
"""

p_sql_answer_fetaqa_with_summary = """
You will receive the **full table, the question, and the summary**.
Generate SQL with a detailed explanation and the final query.  
Exclude irrelevant rows (e.g., "Total", "Average", "World", "N/A", "?", "—") if the summary marks them as such.  

**Important Rule:**
- When using COUNT, you must **never use DISTINCT**.
- Whenever a calculation (e.g., SUM, AVG, MAX, MIN, arithmetic comparisons) is required, always CAST the column values to INTEGER or FLOAT in SQL.  
  Example: `SUM(CAST("Total Wins" AS INTEGER))`, `SUM(CAST("Total Wins" AS FLOAT))`, 
- Never use `HAVING` for conditions on zero aggregrates
- All missing values in the table are stored as NULL
- When filtering categorical values that might appear as part of a longer string 
  (e.g., "No playoff" vs "Champion (no playoff)"), always use a `LIKE '%value%'` condition instead of `=`.

Always **read the summary first** and start with an explanation line beginning with 'Explanation:'.  
- If the summary labels a row as "ordinary", you must not exclude it.  
- If the summary marks a row as an outlier and it is relevant to the question, explicitly exclude or handle it in your SQL.  
- In the explanation, explicitly mention which columns and values from the summary are relevant to answer the question.  
- You must use the provided summary carefully: if the summary specifies data types (e.g., Numeric, Categorical) or shows example values, use this information to decide how to construct conditions or comparisons in SQL.   
- Outlier notes in the summary must also be considered: explain whether they should be excluded from the SQL.  
- Never ignore the summary: every reasoning step must be grounded in it.  

**If the summary does not mention a column or entity explicitly, but the question specifies a clear condition (e.g., a player name, a competition, a year), you must use that condition directly in the SQL filter based on the most relevant column from the full table preview.**

After the explanation, on the next line, output the SQL query prefixed with 'SQL:'.  
- Always wrap column names in double quotes (e.g., SELECT "column_name" FROM T).  
- The SQL must directly follow from the reasoning in the Explanation section.  

If the SQL alone cannot fully answer the question, state in the Explanation how the summary information should be used alongside SQL execution to derive the final answer.  

Your final output must always include **both** the Explanation and the SQL, in that order.

Examples :
SQLite table properties:

Table: List of Olympic Events (row_number, year, sport, event, athlete, medal)

Q: how many events were held in 2008?
Explanation: The summary indicates that the "year" column is numeric, and the "event" column is categorical with each row representing a single Olympic event.
To answer the question, we count the total number of rows where "year" = 2008.
It is important to note that we must count all rows, not just distinct events, because each row already corresponds to one event. Using DISTINCT here would undercount the total.
SQL: select count("event") from T where cast("year" as integer) = 2008;

SQLite table properties:

Table: List of storms on the Great Lakes(row_number,ship,type_of_vessel,lake,location,lives_lost)

Q: how many more ships were wrecked in lake huron than in erie?
Explanation: The summary indicates that the **"lake" column** contains categorical values such as Lake Huron and Lake Erie.  
To answer the question, we count the number of rows where "lake" = Lake Huron and subtract the number of rows where "lake" = Lake Erie.  
No placeholder or aggregate rows (e.g., "Total", "—", "N/A") are present in the summary, so all rows are valid for counting.  
SQL: select ((select count("ship") from T where "lake" like '%lake huron%') - (select count("ship") from T where "lake" like '%lake erie%'))

SQLite table properties:

Table: List of The Citadel Bulldogs football seasons (row_number, season, conference, head coach, total wins, total losses, total ties, conference wins, conference losses, conference ties, conference standing, postseason result)

Q: how many total wins did the citadel bulldogs have before 1908?
Explanation: The summary specifies that the "season" column is Numeric and can be used to filter rows. 
The "total wins" column is also Numeric, but outlier rows (e.g., “Totals: 105 Seasons …”) must be excluded because they represent cumulative statistics rather than a single season. 
To compute the answer, we filter rows where season ≤ 1907 and sum the values in "total wins". 
Since numeric columns may contain non-numeric placeholders in other contexts, we explicitly cast "total wins" as INTEGER.
SQL:
select sum(cast("total wins" as integer)) as total_wins from T where cast("season" as integer) <= 1907;


SQLite table properties:

Table: List of Olympic medal counts (row_number, country, gold, silver, bronze, total)

Q: what is the total number of medals awarded in the olympics?
Explanation: The summary explicitly marks the row "World Total" as an aggregate row.
Since the question asks for the total number of medals across all countries, we should use the aggregate row directly instead of summing all individual country rows (to avoid double counting).
We select the value from the "total" column where "country" = 'World Total'".
SQL:
select cast("total" as integer) as global_total_medals from T where "country" = 'World Total';
"""


def truncate_tokens(prompt,  max_length) -> str:
    """Truncates a text string based on max number of tokens."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encoded_string = encoding.encode(prompt)
    num_tokens = len(encoded_string)

    if num_tokens > max_length:
        prompt = encoding.decode(encoded_string[:max_length])
        print('truncated -->  ', num_tokens)
    return prompt

def get_completion(prompt, model="gpt-4o", temperature=0.2, n=1):
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
    prompt = "" + p_sql_answer_fetaqa_with_summary

    prompt += "\nSQLite table properties:\n\n"
    prompt += "Table: " + title + " (" + str(tab_col) + ")" + "\n\n"

    # ✅ Full Table Preview 추가
    prompt += "Full Table Preview:\n"
    prompt += truncate_tokens(full_table, max_length=15000) + "\n\n"

    if summary:  # ✅ 요약문이 있으면 반드시 포함
        prompt += "Structured table summary:\n" + summary + "\n\n"

    prompt += "Q: " + question + "\n"
    prompt += "Explanation:"
    prompt += "SQL:"
    return prompt

def generate_sql_answer_prompt(title, sql, result_table, question):
    prompt = p_answer_fetaqa
    prompt += "\nTable_title: " + title
    prompt += "\nSQL: " + sql
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

    prompt = p_fetaqa_full
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