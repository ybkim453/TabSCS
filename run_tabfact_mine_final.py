# === Final: Sub-table + Summary + SQL Reasoning + TabFact ===
import re
import os
import csv
import json
import sqlite3
import pandas as pd
from io import StringIO
from openai import OpenAI

from utils.preprocess import *
from utils.prompt_tabfact import *
from subtable_extractor_euclid import ColumnSimilarityAnalyzer

# OpenAI client
client = OpenAI()
analyzer = ColumnSimilarityAnalyzer()

def extract_sql_only(text: str) -> str:
    lines = text.splitlines()
    sql_lines = []
    capture = False
    for line in lines:
        if line.strip().lower().startswith("sql:"):
            part = line.split("SQL:", 1)[1].strip()
            if part:
                sql_lines.append(part)
            capture = True
            continue
        if capture:
            sql_lines.append(line.strip())
    return " ".join(sql_lines).strip()

def parse_tabfact_label(response: str) -> int:
    resp = response.lower()
    if any(x in resp for x in ["not possible", "cannot be verified", "no information", "cannot be determined"]):
        return 2
    elif "true" in resp or "support" in resp:
        return 1
    elif "false" in resp or "refute" in resp:
        return 0
    else:
        return 3

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

def extract_numeric_columns(summary_text: str):
    numeric_cols = []
    if "**Column Analysis:**" in summary_text:
        col_block = summary_text.split("**Column Analysis:**")[1]
        if "**Outlier Row Analysis:**" in col_block:
            col_block = col_block.split("**Outlier Row Analysis:**")[0]

        pattern = r"\d+\.\s+(.*?)\n\s+- Data type:\s+(\w+)"
        matches = re.findall(pattern, col_block)

        for col_name, dtype in matches:
            if dtype.lower() == "numeric":
                numeric_cols.append(col_name.strip())
    return numeric_cols

def normalize_numeric(x):
    if x is None or str(x).lower() in ["nan", "none", ""]:
        return None
    s = str(x).strip()
    s = re.sub(r"[£$€,]", "", s)  # 화폐/기호 제거
    match = re.search(r"-?\d+(\.\d+)?", s)
    if match:
        try:
            val = float(match.group())
            return int(val) if val.is_integer() else val
        except:
            return s
    return s


def tabsqlify_tabfact(T, title, tab_col, statement, full_table, summary,
                      log_path=None, golden_label=None, preview_table=None, table_id=None):

    response = ""
    predict = 3
    result_sql = pd.DataFrame()

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

    # 초기 정보 기록
    log("=== Table ID ===")
    log(str(table_id))
    log("\n=== Statement ===")
    log(statement)
    log("\n=== Sub-table Preview ===")
    log(preview_table)
    log("\n=== Table Summary ===")
    log(summary)
    log("\n=== Full Table Preview ===")
    log(T.to_markdown(index=False))
    
    # SQL 생성
    prompt_sql = gen_table_decom_prompt(title, tab_col, statement, full_table, summary=summary)
    sql_final = get_sql_3(prompt_sql)
    sql_final = extract_sql_only(sql_final)
    log("\nGenerated Final SQL:")
    log(sql_final)

    try:
        result_sql = pd.read_sql_query(sql_final, conn)
        log(f"\nSQL 실행 결과 셀 수: {result_sql.size}")
        if not result_sql.empty:
            log("=== SQL 실행 결과 미리보기 ===")
            log(result_sql.to_markdown(index=False))
    except Exception as e:
        log(f"[Error] Final SQL 실행 실패: {e}")
        result_sql = None   # ❌ SQL 자체가 실패했을 때만 None 처리

    # Reasoning
    if result_sql is not None:
        # ✅ SQL이 정상 실행된 경우 (empty여도 처리)
        linear_table = table_linearization(result_sql, style="pipe")
        reasoning_prompt = generate_sql_answer_prompt(title, sql_final, linear_table, statement)
        log("\nReasoning Prompt:")
        log(reasoning_prompt)

        response = get_answer(reasoning_prompt)
        log("\nRaw Response:")
        log(response.strip())

        predict = parse_tabfact_label(response)
        log(f"Prediction: {predict}")
        log(f"[Gold Label] {golden_label}\n")

    else:
        # ✅ SQL 실행 자체가 실패했을 때만 fallback
        log("[Fallback] SQL 실행 실패 → Full Table 기반 추론")
        linear_table = table_linearization(T, style="pipe")
        prompt_ans = gen_full_table_prompt(title, linear_table, statement)
        log("\n[Fallback] Prompt SQL:")
        log(prompt_ans)

        response = get_answer(prompt_ans)
        log("[Fallback] Raw Response:")
        log(response.strip())

        predict = parse_tabfact_label(response)
        log(f"[Fallback] Prediction: {predict}")
        log(f"[Gold Label] {golden_label}\n")


    conn.close()
    if flog: flog.close()
    return sql_final, result_sql, response, predict, summary

if __name__ == "__main__":
    path = "datasets/tabfact_small_test.jsonl"
    base_output = "outputs_tabfact"
    subtable_dir = os.path.join(base_output, "subtables")
    summary_dir = os.path.join(base_output, "summary_log")
    sql_log_dir = os.path.join(base_output, "sql_logs")

    os.makedirs(base_output, exist_ok=True)
    os.makedirs(subtable_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(sql_log_dir, exist_ok=True)

    # ✅ 실행 범위 지정
    start = 3
    end = 4   # 예: 처음 100개만 실행

    correct, wrong, total = 0, 0, 0

    with open(path, encoding="utf-8") as f1, \
         open(os.path.join(base_output, "tabfact_results.jsonl"), "a", encoding="utf-8") as fw, \
         open(os.path.join(base_output, "tabfact_results.csv"), "a", newline="", encoding="utf-8") as fcsv:

        writer = csv.writer(fcsv)
        header = ["id", "statement", "label", "prediction", "sql", "response", "summary"]
        writer.writerow(header)

        for i, l in enumerate(f1):
            if i < start or i >= end:
                continue   # ✅ 범위 벗어나면 skip
            dic = json.loads(l)
            tid = dic["table_id"]
            title = dic["table_caption"]
            statement = dic["statement"]
            label = dic["label"]

            T = dict2df(dic["table_text"])
            T = T.assign(row_number=range(len(T)))
            row_number = T.pop("row_number")
            T.insert(0, "row_number", row_number)

            tab_col = ", ".join(T.columns)
            full_table = table_linearization(T, style="pipe")

            # === Sub-table 저장 (캐시) ===
            subtable_path = os.path.join(subtable_dir, f"{tid.replace('/','_')}.tsv")
            if os.path.exists(subtable_path):
                with open(subtable_path, "r", encoding="utf-8") as f:
                    result_lines = f.read().splitlines()
            else:
                result_lines = T.head(5).to_csv(sep="\t", index=False).splitlines()
                with open(subtable_path, "w", encoding="utf-8") as f:
                    for line in result_lines:
                        f.write(line + "\n")
            subtable_df = pd.read_csv(StringIO("\n".join(result_lines)), sep="\t")
            preview_table = subtable_df.to_markdown(index=False)

            # === Summary 저장 (캐시) ===
            summary_path = os.path.join(summary_dir, f"{tid.replace('/','_')}_summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = f.read().strip()
            else:
                summary = call_gpt_table_summary(preview_table, [])
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)

            # === SQL Reasoning ===
            sql, result, response, predict, summary = tabsqlify_tabfact(
                T, title, tab_col, statement, full_table, summary,
                log_path=os.path.join(sql_log_dir, f"{i}.txt"),
                golden_label=label,
                preview_table=preview_table,
                table_id=tid
            )

            if predict == label:
                correct += 1
            else:
                wrong += 1
            total += 1

            record = {
            "idx": i,
            "prediction": predict,
            "label": label,
            "response": response,
            "statement": statement,
            "key": tid
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Final Accuracy: {correct}/{total} ({correct/(total+1e-4):.4f})")
