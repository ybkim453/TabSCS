# === Final: FeTaQA Sub-table + Summary + SQL Reasoning + Natural-language Answer Generation ===
import re
import os
import csv
import json
import sqlite3
import pandas as pd
from io import StringIO
from openai import OpenAI

from utils.preprocess import *
from utils.prompt_fetaqa import *
from subtable_extractor_euclid import ColumnSimilarityAnalyzer

# ---------------------- Client & Analyzer ----------------------
client = OpenAI()
analyzer = ColumnSimilarityAnalyzer()

# ---------------------- 공통 유틸 ----------------------
def extract_sql_only(text: str) -> str:
    """LLM 생성 결과에서 SQL 부분만 추출"""
    lines = text.splitlines()
    sql_lines = []
    capture = False
    for line in lines:
        line_strip = line.strip()
        lower_line = line_strip.lower()

        # SQL 시작 감지 (대소문자 모두 허용)
        if lower_line.startswith("sql:"):
            parts = re.split(r"sql\s*:", line_strip, flags=re.IGNORECASE)
            if len(parts) > 1:
                sql_lines.append(parts[1].strip())
            capture = True
            continue

        # 코드 블록 시작 감지
        if lower_line.startswith("```sql") or lower_line.startswith("```"):
            capture = True
            continue

        # 코드 블록 종료 감지
        if lower_line.startswith("```") and capture:
            break

        if capture:
            sql_lines.append(line_strip)

    # SQL 문자열 결합 및 정제
    sql = " ".join(sql_lines)
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    sql = sql.strip('`"\' ')
    return sql

def parse_answer(response: str) -> str:
    """LLM 응답에서 Final Answer 부분만 추출"""
    output_ans = response
    try:
        output_ans = response.split("Final Answer:")[1]
    except Exception:
        try:
            output_ans = response.split("Answer:")[1]
        except Exception:
            pass
    return output_ans.strip().lower()

def call_gpt_table_summary(table_markdown: str, outliers: list, model: str = "gpt-3.5-turbo") -> str:
    prompt = f"""
You are a strict table analysis assistant.
Here is a partial preview of a table in markdown format:

{table_markdown}

Additionally, the following ROW INDICES were identified as OUTLIERS (atypical rows) by statistical similarity analysis:
{json.dumps(outliers, ensure_ascii=False)}

**Important Rules:**
- Not all candidate outlier rows are truly special. Some are just ordinary data rows.
- Do not treat empty or missing cells as distinctive values. They are simple missing data and must not be treated as distinctive outliers.

**Your task:**
1. For every column, describe:
   - Data type (numeric, categorical, etc.)
   - 4 representative example values
   - Role of the column (identifier, descriptor, measure, etc.)

2. Outlier Row Analysis:
- For each candidate row, explicitly mention the exact values that make it distinctive (e.g., “National Cup = Semifinals”, “Reg. Season = 5th?”).
- If these values are clearly different from the majority of the table, explain why this makes the row an outlier.
- If the values are ambiguous markers like "?" or "5th?" but are ordinary within the context of the table, state explicitly that they are ordinary values and not special.
- If the row is consistent with ordinary entries, state that it is an ordinary row and requires no special handling.
- Focus only on candidate rows; do not discuss rows outside the candidate list.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# ---------------------- SQL Reasoning ----------------------
def tabsqlify_fetaqa(T, title, tab_col, question, full_table, summary, 
                     log_path=None, golden_answer=None, preview_table=None, table_id=None):
    """FeTaQA용 SQL reasoning + 자연어 답변 생성"""
    response = ""
    output_ans = ""
    linear_table = ""
    result_sql = pd.DataFrame()

    # === SQLite 메모리 DB 연결 ===
    conn = sqlite3.connect(":memory:")
    T.to_sql("T", conn, index=False, if_exists="replace")

    # === 로그 준비 ===
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        flog = open(log_path, "w", encoding="utf-8")
    else:
        flog = None

    def log(msg):
        print(msg)
        if flog:
            flog.write(msg + "\n")

    if flog:
        flog.write(f"=== Table ID: {table_id} ===\n")
        flog.write(f"=== Question: {question} ===\n\n")
        flog.write("=== Sub-table Preview ===\n")
        flog.write(preview_table + "\n\n")
        flog.write("=== Table Summary ===\n")
        flog.write(summary + "\n\n")
        flog.write("=== Full Table Preview ===\n")
        flog.write(T.to_markdown(index=False) + "\n\n")
        flog.flush()

    # === SQL 생성 프롬프트 ===
    prompt_sql = gen_table_decom_prompt(title, tab_col, question, full_table, summary=summary)
    sql_final = get_sql_3(prompt_sql)
    log(f"Generated Reasoning SQL: {sql_final}")
    sql_final = extract_sql_only(sql_final)
    log(f"Extracted Final SQL: {sql_final}")

    # === SQL 실행 ===
    try:
        result_sql = pd.read_sql_query(sql_final, conn)
        log(f"SQL 실행 결과 셀 수: {result_sql.size}")
        if not result_sql.empty:
            log("=== SQL 실행 결과 미리보기 ===")
            log(result_sql.to_markdown(index=False))
    except Exception as e:
        log(f"[Error] SQL 실행 실패: {e}")
        result_sql = pd.DataFrame()

    # === SQL 결과가 존재할 경우 자연어 Reasoning ===
    if not result_sql.empty:
        linear_table = table_linearization(result_sql, style='pipe')
        reasoning_prompt = generate_sql_answer_prompt(title, sql_final, linear_table, question)
        log(f"Reasoning Prompt:\n{reasoning_prompt}")

        response = get_answer(reasoning_prompt)
        output_ans = parse_answer(response)
        log(f"Prediction: {output_ans}")
        log(f"[Gold Answer] {golden_answer}\n")

        if flog: flog.close()
        conn.close()
        return sql_final, result_sql, response, output_ans, linear_table

    # === SQL 실패 시 Full Table 기반 Fallback ===
    log("[Fallback] SQL 결과가 없으므로 Full Table 기반 추론으로 전환")
    sql = "select * from T"
    result = T.copy()
    linear_table = table_linearization(result, style='pipe')
    prompt_ans = gen_full_table_prompt(title, tab_col, linear_table, question)

    response = get_answer(prompt_ans)
    output_ans = parse_answer(response)

    log(f"[Fallback] Prediction: {output_ans}")
    log(f"[Gold Answer] {golden_answer}\n")

    if flog: flog.close()
    return sql, result, response, output_ans, linear_table


# ---------------------- 메인 실행 ----------------------
if __name__ == "__main__":
    path = 'datasets/fetaQA-v1_test.jsonl'
    start = 0
    end = 2000  # 처리 범위 조정

    base_output = "outputs_fetaqa"
    subtable_dir = os.path.join(base_output, "subtables")
    summary_dir = os.path.join(base_output, "summary_log")
    sql_log_dir = os.path.join(base_output, "sql_logs")
    os.makedirs(base_output, exist_ok=True)
    os.makedirs(subtable_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(sql_log_dir, exist_ok=True)

    correct = 0
    total = 0

    with open(path, encoding='utf-8') as f1, \
         open(os.path.join(base_output, 'fetaqa_results.jsonl'), 'a', encoding='utf-8') as fw, \
         open(os.path.join(base_output, 'fetaqa_results.csv'), 'a', newline='', encoding='utf-8') as fcsv:

        writer = csv.writer(fcsv)
        header = ['id', 'question', 'answer', 'prediction', 'sql', 'response',
                  'summary', 'r_num_cell', 't_num_cell']
        writer.writerow(header)

        for i, l in enumerate(f1):
            if i < start or i >= end:
                continue

            dic = json.loads(l)
            feta_id = dic.get('feta_id', f"id_{i}")
            title = dic['table_page_title']
            question = dic['question']
            answer = dic['answer'].lower()
            table = dic['table_array']

            print(f"\n=== ID: {feta_id}, Q: {question}, Gold: {answer} ===")

            # === 1) Table Load ===
            T = dict2df(table)
            T = T.assign(row_number=range(len(T)))
            row_number = T.pop('row_number')
            T.insert(0, 'row_number', row_number)
            tab_col = ", ".join(T.columns)

            # === 2) Sub-table 생성 ===
            csv_content = "\n".join(["\t".join(map(str, row)) for row in table])
            subtable_df = pd.read_csv(StringIO(csv_content), sep="\t", header=None)
            preview_table = subtable_df.to_markdown(index=False)

            # === 3) Summary 생성 (혹은 캐싱) ===
            summary_path = os.path.join(summary_dir, f"{feta_id}_summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = f.read().strip()
            else:
                outliers = []
                summary = call_gpt_table_summary(preview_table, outliers)
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)

            full_table = table_linearization(T, style='pipe')

            # === 4) SQL + Reasoning ===
            sql, result, response, output_ans, linear_table = tabsqlify_fetaqa(
                T, title, tab_col, question, full_table, summary,
                log_path=os.path.join(sql_log_dir, f"{feta_id}.txt"),
                golden_answer=answer,
                preview_table=preview_table,
                table_id=feta_id
            )

            # === 5) 평가 ===
            output_ans = output_ans.lower()
            if output_ans.strip() == answer or \
               output_ans.strip().find(answer) != -1 or \
               answer.strip().find(output_ans.strip()) != -1:
                correct += 1
            total += 1

            acc = correct / (total + 1e-5)
            print(f"[Prediction] {output_ans} | [Gold] {answer} | Acc={acc:.4f}")

            # === 6) 저장 ===
            tmp = {
                'idx': i,
                'feta_id': feta_id,
                'prediction': output_ans,
                'answer': answer,
                'question': question,
                'full_response': response,
            }
            fw.write(json.dumps(tmp) + '\n')

            data = [feta_id, question, answer, output_ans, sql, response,
                    summary, result.size, T.size]
            writer.writerow(data)

    print(f"\n✅ Final Accuracy: {correct}/{total} ({correct / (total + 1e-5):.4f})")
