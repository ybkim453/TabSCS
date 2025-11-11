# TabBridge: Table Reasoning with Evaluation-Feedback-Regeneration Loop
import re
import os
import csv
import json
import sqlite3
import pandas as pd
from io import StringIO
from openai import OpenAI

from utils.preprocess import *
from utils.prompt_wtq import *
from utils.ss_row_analysis_evaluator import SSRowAnalysisEvaluator
from utils.ss_reconstruction_based_evaluator import SSReconstructionBasedEvaluator
from utils.sql_evaluator import SQLEvaluator
from subtable_extractor_euclid import ColumnSimilarityAnalyzer

def load_prompt(filename):
    """Load prompt file from prompt directory"""
    prompt_dir = "prompt"
    file_path = os.path.join(prompt_dir, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file {filename} not found. Using fallback.")
        return ""

def generate_ss_feedback(row_eval, recon_eval, preview_table, current_summary, subtable_df, generated_table_text=None):
    """Generate feedback based on SS evaluation results""" 
    feedbacks = []
    
    if row_eval and not row_eval.get('all_criteria_passed', False):
        row_feedback_prompt = load_prompt("SS_feedback/row_analysis_feedback.txt")
        if row_feedback_prompt:
            evaluation_reasoning = ""
            
            if 'row_analysis_discrimination' in row_eval:
                evaluation_reasoning += f"Row Analysis Discrimination: {row_eval['row_analysis_discrimination'].get('reasoning', '')}\n"
            if 'accuracy_of_classification' in row_eval:
                evaluation_reasoning += f"Accuracy of Classification: {row_eval['accuracy_of_classification'].get('reasoning', '')}"
            
            prompt = row_feedback_prompt.replace(
                "{sub_table}", preview_table
            ).replace(
                "{row_analysis}", current_summary
            ).replace(
                "{evaluation_reasoning}", evaluation_reasoning
            )
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                feedbacks.append(f"Row Analysis Feedback: {response.choices[0].message.content}")
            except Exception as e:
                print(f"Error generating row analysis feedback: {e}")
    
    if recon_eval and recon_eval.get('header_em_score', 0) < 1.0:
        header_feedback_prompt = load_prompt("SS_feedback/header_feedback.txt")
        if header_feedback_prompt and generated_table_text:
            prompt = header_feedback_prompt.replace(
                "{sub_tab}", preview_table
            ).replace(
                "{generated_table}", generated_table_text
            ).replace(
                "{em_score}", str(recon_eval.get('header_em_score', 0))
            )
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                feedbacks.append(f"Header Feedback: {response.choices[0].message.content}")
            except Exception as e:
                print(f"Error generating header feedback: {e}")
    
    if recon_eval and recon_eval.get('cell_similarity_score', 0) < 0.85:
        structure_feedback_prompt = load_prompt("SS_feedback/structure_feedback.txt")
        if structure_feedback_prompt and generated_table_text:
            prompt = structure_feedback_prompt.replace(
                "{sub_table}", preview_table
            ).replace(
                "{generated_table}", generated_table_text
            ).replace(
                "{bert_score}", str(recon_eval.get('cell_similarity_score', 0))
            )
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                feedbacks.append(f"Structure Feedback: {response.choices[0].message.content}")
            except Exception as e:
                print(f"Error generating structure feedback: {e}")
    
    return "\n\n".join(feedbacks) if feedbacks else ""

def refine_ss_with_feedback(original_ss, feedback, preview_table, outliers):
    """Refine SS based on feedback"""
    if not feedback:
        return original_ss
    
    refinement_prompt = f"""You are an expert at improving table analysis based on feedback.

**Original SS:**
{original_ss}

**Feedback for Improvement:**
{feedback}

**Sub-table:**
{preview_table}

**Outlier Candidates:**
{json.dumps(outliers, ensure_ascii=False)}

**Task:**
Based on the feedback provided, generate an improved SS that addresses all the issues mentioned. 
Focus on:
1. Correcting any header extraction or naming issues
2. Improving row analysis discrimination
3. Enhancing content accuracy and semantic understanding

Generate the improved SS following the same format as the original."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": refinement_prompt}],
            temperature=0.2,
            max_tokens=3000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error refining SS with feedback: {e}")
        return original_ss

# OpenAI client
client = OpenAI()
analyzer = ColumnSimilarityAnalyzer()

row_evaluator = SSRowAnalysisEvaluator()
recon_evaluator = SSReconstructionBasedEvaluator()

sql_evaluator = SQLEvaluator()

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

def parse_answer(response: str) -> str:
    output_ans = response
    try:
        output_ans = response.split("Answer:")[1]
    except:
        output_ans = "" + response
    match = re.search(r'(The|the) answer is ([^\.]+)\.$', output_ans)
    if match:
        output_ans = match.group(2).strip('"')
    return output_ans.strip().lower()


def call_gpt_table_summary(table_markdown: str, outliers: list, model: str = "gpt-3.5-turbo") -> str:
    ss_generation_template = load_prompt("SS_Generation.txt")
    
    if not ss_generation_template:
        ss_generation_template = """You are a strict table analysis assistant.
Here is a partial preview of a table in markdown format:

{table_markdown}

Additionally, the following ROW INDICES were identified as ATYPICAL ANALYSIS by statistical similarity analysis:
{json.dumps(special_rows, ensure_ascii=False)}

**Important Rules:**
- Not all candidate row analysis are truly special. Some are just representative data rows.
- Do not treat empty or missing cells as distinctive values. They are simple missing data and must not be treated as distinctive special row.

**Your task:**
1. For every column, describe:
   - Data type (numeric, categorical, etc.)
   - 4 representative example values
   - Role of the column (identifier, descriptor, measure, etc.)

2. Row Analysis:
- For each candidate row, explicitly mention the exact values that make it distinctive (e.g., "National Cup = Semifinals", "Reg. Season = 5th?").
- If these values are clearly different from the majority of the table, explain why this makes the row an special row.
- Focus only on candidate rows; do not discuss rows outside the candidate list."""
    
    prompt = ss_generation_template.format(
        table_markdown=table_markdown,
        special_rows=outliers
    ).replace("{json.dumps(special rows, ensure_ascii=False)}", json.dumps(outliers, ensure_ascii=False))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------- SQL Reasoning ----------------------
def tabsqlify_wtq(T, title, tab_col, question, full_table, summary, 
                  log_path=None, golden_answer=None, preview_table=None, table_id=None) :
    """col 기반 → sql reasoning → fallback, 모든 과정 로그 저장"""
    response = ""
    output_ans = ""
    linear_table = ""
    result_sql = pd.DataFrame()

    conn = sqlite3.connect(":memory:")
    T.to_sql("T", conn, index=False, if_exists="replace")

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
        flog.write("=== Table ID ===\n")
        flog.write(str(dic.get("table_id", "UNKNOWN")) + "\n\n")
        flog.write("=== Question ===\n")
        flog.write(question + "\n\n")
        flog.write("=== Sub-table Preview ===\n")
        flog.write(preview_table + "\n\n")
        flog.write("=== Table Summary ===\n")
        flog.write(summary + "\n\n")
        flog.write("=== Full Table Preview ===\n")
        flog.write(T.to_markdown(index=False) + "\n\n")
        flog.flush()

    sql_final = None
    for sql_attempt in range(3):
        log(f"SQL Generation Attempt {sql_attempt+1}/3")
        
        prompt_sql = gen_table_decom_prompt(title, tab_col, question, full_table, summary = summary)
        current_sql_response = get_sql_3(prompt_sql)
        log(f"Generated Reasoning SQL: {current_sql_response}")
        current_sql = extract_sql_only(current_sql_response) 
        log(f"Generated Final SQL: {current_sql}")
        
        log(f"  Evaluating SQL quality...")
        sql_evaluation = sql_evaluator.evaluate_sql_query(
            question, full_table, current_sql, summary,
            log_file_path=log_path.replace('.txt', f'_sql_eval_{sql_attempt+1}.txt') if log_path else None
        )
        
        sql_passed = sql_evaluation.get('all_criteria_passed', False) if sql_evaluation else False
        log(f"  SQL Evaluation: {'PASS' if sql_passed else 'FAIL'}")
        
        if sql_evaluation:
            criteria = ['Appropriate Role', 'Well Used Row Analysis', 'Appropriate Data Type', 'Faithfulness']
            for criterion in criteria:
                result = sql_evaluation.get(criterion, {}).get('result', 'unknown')
                log(f"    {criterion}: {result.upper()}")
        
        if log_path:
            eval_summary_path = log_path.replace('.txt', f'_sql_evaluation_summary_{sql_attempt+1}.txt')
            with open(eval_summary_path, "w", encoding="utf-8") as f:
                f.write(f"=== SQL Evaluation Summary - Attempt {sql_attempt+1} ===\n\n")
                f.write(f"Question: {question}\n\n")
                f.write(f"Generated SQL: {current_sql}\n\n")
                f.write(f"SQL Passed: {sql_passed}\n\n")
                if sql_evaluation:
                    f.write("Evaluation Details:\n")
                    for criterion in criteria:
                        result = sql_evaluation.get(criterion, {}).get('result', 'unknown')
                        reasoning = sql_evaluation.get(criterion, {}).get('reasoning', 'No reasoning provided')
                        f.write(f"  {criterion}: {result.upper()}\n")
                        f.write(f"    Reasoning: {reasoning}\n\n")
        
        if sql_passed:
            log(f"SQL evaluation passed - Attempt {sql_attempt+1}/3")
            sql_final = current_sql
            break
        else:
            if sql_attempt < 2:
                log(f"SQL evaluation failed - generating feedback and refinement")
                
                feedback = sql_evaluator.generate_sql_feedback(
                    question, summary, current_sql, sql_evaluation
                )
                
                if feedback:
                    log(f"    Generated SQL feedback: {feedback[:200]}...")
                    
                    if log_path:
                        feedback_path = log_path.replace('.txt', f'_sql_feedback_{sql_attempt+1}.txt')
                        with open(feedback_path, "w", encoding="utf-8") as f:
                            f.write(f"=== SQL Feedback - Attempt {sql_attempt+1} ===\n\n")
                            f.write(f"Question: {question}\n\n")
                            f.write(f"Generated SQL: {current_sql}\n\n")
                            f.write(f"SQL Evaluation Result:\n{json.dumps(sql_evaluation, indent=2, ensure_ascii=False)}\n\n")
                            f.write(f"Generated Feedback:\n{feedback}\n")
                    
                    refined_sql = sql_evaluator.refine_sql_with_feedback(
                        current_sql, feedback, question, summary, full_table
                    )
                    log(f"    SQL refined based on feedback: {refined_sql}")
                    
                    if log_path:
                        refined_sql_path = log_path.replace('.txt', f'_sql_refined_{sql_attempt+1}.txt')
                        with open(refined_sql_path, "w", encoding="utf-8") as f:
                            f.write(f"=== Refined SQL - Attempt {sql_attempt+1} ===\n\n")
                            f.write(f"Original SQL: {current_sql}\n\n")
                            f.write(f"Feedback: {feedback}\n\n")
                            f.write(f"Refined SQL: {refined_sql}\n")
                else:
                    log(f"    No specific SQL feedback generated")
            else:
                log(f"Failed after 3 attempts - using last SQL")
                sql_final = current_sql

    try:
        result_sql = pd.read_sql_query(sql_final, conn)
        log(f"SQL execution result cell count: {result_sql.size}")
        if not result_sql.empty:
            log("=== SQL Execution Result Preview ===")
            log(result_sql.to_markdown(index=False))
    except Exception as e:
        log(f"[Error] Final SQL execution failed: {e}")
        result_sql = pd.DataFrame()
    
    if not result_sql.empty:
        if result_sql.isnull().all().all():
            log("[Warning] SQL result contains only None/NaN - switching to fallback")
            result_sql = pd.DataFrame()
        else:
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

    log("[Fallback] No SQL result, switching to full table-based reasoning")
    sql = "select * from T"
    result = T.copy() # copy full table
    linear_table = table_linearization(result, style='pipe')
    prompt_ans = gen_full_table_prompt(title, tab_col, linear_table, question)

    log(f"[Fallback] Prompt SQL:\n{prompt_ans}")
    response = get_answer(prompt_ans)
    log(f"[Fallback] Raw Response: {response}")

    output_ans = parse_answer(response)
    log(f"[Fallback] Prediction: {output_ans}")
    log(f"[Gold Answer] {golden_answer}\n")

    if flog: flog.close()
    return sql, result, response, output_ans, linear_table


if __name__ == "__main__":
    path = 'datasets/wtq.jsonl'
    start = 0
    end = 1

    table_ids = list(range(start, end))
    base_output = "outputs"
    subtable_dir = os.path.join(base_output, "subtables")
    ss_logs_dir = os.path.join(base_output, "ss_logs")
    sql_logs_dir = os.path.join(base_output, "sql_logs")

    os.makedirs(base_output, exist_ok=True)
    os.makedirs(subtable_dir, exist_ok=True)
    os.makedirs(ss_logs_dir, exist_ok=True)
    os.makedirs(sql_logs_dir, exist_ok=True)

    correct = 0
    t_samples = 0

    with open(path, encoding='utf-8') as f1, \
         open(os.path.join(base_output, 'wtq_results.jsonl'), 'a', encoding='utf-8') as fw, \
         open(os.path.join(base_output, 'wtq_results.csv'), 'a', newline='', encoding='utf-8') as fcsv:

        writer = csv.writer(fcsv)
        header = ['id', 'question', 'answer', 'prediction', 'sql', 'response',
                  'summary', 'r_num_cell', 't_num_cell']
        writer.writerow(header)

        for i, l in enumerate(f1):
            if i in table_ids:
                dic = json.loads(l)
                idx = dic['id']
                title = dic['title']
                question = dic['question']
                answer = ','.join(dic['answer']).lower()
                table_id = dic['table_id']

                print(f"\n=== ID: {idx}, Q: {question}, Gold: {answer} ===")

                subtable_path = os.path.join(subtable_dir, f"{table_id.replace('/','_')}.tsv")
                outliers = []

                if os.path.exists(subtable_path):
                    print(f"[Cache Hit] Using cached subtable: {subtable_path}")
                    with open(subtable_path, "r", encoding="utf-8") as f:
                        result_lines = f.read().splitlines()
                else:
                    print(f"[Cache Miss] Creating subtable for {table_id}")
                    result_lines, selected_rows, outliers = analyzer.process_jsonl_record(
                        jsonl_path=path,
                        index=i,
                        log_file_path=os.path.join(subtable_dir, f"{table_id.replace('/','_')}_log.txt")
                    )
                    with open(subtable_path, "w", encoding="utf-8") as f:
                        for line in result_lines:
                            f.write(line + "\n")

                csv_content = "\n".join(result_lines)
                subtable_df = pd.read_csv(StringIO(csv_content), sep="\t")
                preview_table = subtable_df.to_markdown(index=False)

                summary_path = os.path.join(ss_logs_dir, f"{i}_summary.txt")
                if os.path.exists(summary_path):
                    print(f"[Cache Hit] Using cached summary: {summary_path}")
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = f.read().strip()
                else:
                    print(f"[Cache Miss] Creating summary for {table_id}")
                    
                    summary = None
                    for attempt in range(3):
                        print(f"  SS Generation Attempt {attempt+1}/3")
                        
                        current_summary = call_gpt_table_summary(preview_table, outliers)
                        
                        print(f"    Evaluating SS quality...")
                        row_eval = row_evaluator.evaluate_ss_comprehensive(
                            preview_table, outliers, current_summary,
                            log_file_path=os.path.join(ss_logs_dir, f"{i}_row_eval_log_{attempt+1}.txt")
                        )
                        
                        recon_eval = recon_evaluator.evaluate_ss_by_table_generation(
                            title, current_summary, subtable_df,
                            log_file_path=os.path.join(ss_logs_dir, f"{i}_recon_eval_log_{attempt+1}.txt")
                        )
                        
                        row_passed = row_eval['all_criteria_passed'] if row_eval else False
                        recon_passed = recon_eval['evaluation_passed'] if recon_eval else False
                        
                        print(f"    Row Analysis: {'PASS' if row_passed else 'FAIL'}")
                        if recon_eval:
                            print(f"    Reconstruction: {'PASS' if recon_passed else 'FAIL'} (EM: {recon_eval['header_em_score']:.3f}, Sim: {recon_eval['cell_similarity_score']:.3f})")
                        
                        eval_summary_path = os.path.join(ss_logs_dir, f"{i}_ss_evaluation_summary_{attempt+1}.txt")
                        with open(eval_summary_path, "w", encoding="utf-8") as f:
                            f.write(f"=== SS Evaluation Summary - Attempt {attempt+1} ===\n\n")
                            f.write(f"Row Analysis Passed: {row_passed}\n")
                            f.write(f"Reconstruction Passed: {recon_passed}\n")
                            if recon_eval:
                                f.write(f"Header EM Score: {recon_eval['header_em_score']:.4f}\n")
                                f.write(f"Cell Similarity Score: {recon_eval['cell_similarity_score']:.4f}\n")
                            f.write(f"Overall Passed: {row_passed and recon_passed}\n\n")
                            f.write(f"Generated SS:\n{current_summary}\n")
                        
                        if row_passed and recon_passed:
                            print(f"  SS evaluation passed - Attempt {attempt+1}/3")
                            summary = current_summary
                            break
                        else:
                            if attempt < 2:
                                print(f"  SS evaluation failed - generating feedback and refinement")
                                
                                generated_table_text = recon_eval.get('generated_table', '') if recon_eval else ''
                                feedback = generate_ss_feedback(
                                    row_eval, recon_eval, preview_table, 
                                    current_summary, subtable_df, generated_table_text
                                )
                                
                                if feedback:
                                    print(f"    Generated feedback: {feedback[:200]}...")
                                    
                                    feedback_path = os.path.join(ss_logs_dir, f"{i}_ss_feedback_{attempt+1}.txt")
                                    with open(feedback_path, "w", encoding="utf-8") as f:
                                        f.write(f"=== SS Feedback - Attempt {attempt+1} ===\n\n")
                                        f.write(f"Row Evaluation Result:\n{json.dumps(row_eval, indent=2, ensure_ascii=False)}\n\n")
                                        f.write(f"Reconstruction Evaluation Result:\n{json.dumps(recon_eval, indent=2, ensure_ascii=False)}\n\n")
                                        f.write(f"Generated Feedback:\n{feedback}\n\n")
                                        f.write(f"Original SS:\n{current_summary}\n")
                                    
                                    current_summary = refine_ss_with_feedback(
                                        current_summary, feedback, preview_table, outliers
                                    )
                                    print(f"    SS refined based on feedback")
                                    
                                    refined_ss_path = os.path.join(ss_logs_dir, f"{i}_ss_refined_{attempt+1}.txt")
                                    with open(refined_ss_path, "w", encoding="utf-8") as f:
                                        f.write(f"=== Refined SS - Attempt {attempt+1} ===\n\n")
                                        f.write(current_summary)
                                else:
                                    print(f"    No specific feedback generated - using original approach")
                            else:
                                print(f"  Failed after 3 attempts - using last SS")
                                summary = current_summary
                    
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(summary)

                print("\n[Summary]\n", summary)

                # === 3) SQL Reasoning ===
                T = dict2df([dic['table']['header']] + dic['table']['rows'])
                T = T.assign(row_number=range(len(T)))
                row_number = T.pop('row_number')
                T.insert(0, 'row_number', row_number)

                numeric_cols = extract_numeric_columns(summary)

                def normalize_numeric(x):
                    if x is None or str(x).lower() in ["nan", "none", ""]:
                        return None
                    s = str(x).strip()
                    s = re.sub(r"[£$€,]", "", s)
                    match = re.search(r"-?\d+(\.\d+)?", s)
                    if match:
                        try:
                            val = float(match.group())
                            return int(val) if val.is_integer() else val
                        except:
                            return s
                    return s

                for col in T.columns:
                    if col.lower() in [c.lower() for c in numeric_cols]:
                        T[col] = T[col].map(normalize_numeric)

                tab_col = ", ".join(T.columns)
                conn_tmp = sqlite3.connect(":memory:")
                T.to_sql("T", conn_tmp, index=False, if_exists="replace")
                full_table = table_linearization(T, style='pipe')

                sql, result, response, output_ans, linear_table = tabsqlify_wtq(
                    T, title, tab_col, question, full_table, summary,
                    log_path=os.path.join("outputs/sql_logs", f"{i}.txt"),
                    golden_answer=answer,
                    preview_table=preview_table, 
                    table_id=table_id 
                )

                output_ans = output_ans.lower()
                if output_ans.strip() == answer or \
                   output_ans.strip().find(answer) != -1 or \
                   answer.strip().find(output_ans.strip()) != -1:
                    correct += 1

                t_samples += 1
                acc = correct / (t_samples + 0.0001)
                print(f"\n[Prediction] {output_ans} | [Gold] {answer} | Acc={acc:.4f}")

                tmp = {
                    'idx': i,
                    'prediction': output_ans,
                    'answer': answer,
                    'question': question,
                    'response': response,
                    'table_id': table_id,
                    'question_id': idx,
                }
                fw.write(json.dumps(tmp) + '\n')

                data = [idx, question, answer, output_ans.strip(), sql, response,
                        summary, result.size, T.size]
                writer.writerow(data)

    print(f"\nFinal Accuracy: {correct}/{t_samples} ({correct / (t_samples + 0.0001):.4f})")