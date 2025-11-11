"""
SQL Evaluator - SQL 쿼리 평가 및 피드백 생성 모듈
TabBridge에서 사용할 SQL 평가 시스템
"""

import os
import re
import json
from openai import OpenAI

class SQLEvaluator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        """SQL 평가기 초기화"""
        self.client = OpenAI()
        self.model_name = model_name
        self.prompt_dir = "prompt"
    
    def load_prompt(self, filename):
        """프롬프트 파일 로드"""
        file_path = os.path.join(self.prompt_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Prompt file {filename} not found.")
            return ""
    
    def evaluate_sql_query(self, question, table_summary, generated_sql, ss_content, log_file_path=None):
        """SQL 쿼리를 4가지 기준으로 평가"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== SQL Query Evaluation ===")
        log(f"Question: {question}")
        log(f"Generated SQL: {generated_sql}")
        
        # SQL 평가 프롬프트 로드
        evaluation_prompt_path = os.path.join(self.prompt_dir, "SQL_Evaluation.txt")
        try:
            with open(evaluation_prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
        except FileNotFoundError:
            log(f"Failed to load SQL evaluation prompt at {evaluation_prompt_path}")
            return None
        
        # 프롬프트 포맷팅
        prompt = prompt_template.replace(
            "{question}", question
        ).replace(
            "{table}", table_summary
        ).replace(
            "{generated_sql}", generated_sql
        ).replace(
            "{ss}", ss_content
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a strict SQL query evaluation expert. Always return valid JSON following the exact format specified. Evaluate all 4 criteria: Appropriate Role, Well Used Row Analysis, Appropriate Data Type, and Faithfulness."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content
            log(f"LLM Response: {response_text}")
            
            # JSON 추출
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                evaluation_result = json.loads(json_match.group())
                
                # all_criteria_passed 계산 (4개 기준 모두 통과해야 함)
                criteria = ['Appropriate Role', 'Well Used Row Analysis', 'Appropriate Data Type', 'Faithfulness']
                all_passed = all(
                    evaluation_result.get(criterion, {}).get('result', 'no').lower() == 'yes' 
                    for criterion in criteria
                )
                
                evaluation_result['all_criteria_passed'] = all_passed
                
                log(f"Evaluation Results:")
                for criterion in criteria:
                    result = evaluation_result.get(criterion, {}).get('result', 'unknown')
                    log(f"  {criterion}: {result.upper()}")
                log(f"All Criteria Passed: {all_passed}")
                
                return evaluation_result
            else:
                log("No JSON found in SQL evaluation response")
                return None
                
        except Exception as e:
            log(f"Error in SQL evaluation: {str(e)}")
            return None
    
    def generate_sql_feedback(self, question, ss_content, generated_sql, evaluation_result):
        """SQL 평가 결과를 바탕으로 피드백 생성"""
        feedbacks = []
        
        if not evaluation_result:
            return ""
        
        # 각 기준별로 실패한 경우 해당 피드백 생성
        criteria_feedback_mapping = {
            'Appropriate Role': 'appropriate_role.txt',
            'Well Used Row Analysis': 'well_used_row_analysis.txt', 
            'Appropriate Data Type': 'appropriate_data_type.txt',
            'Faithfulness': 'faithfulness.txt'
        }
        
        for criterion, feedback_file in criteria_feedback_mapping.items():
            criterion_result = evaluation_result.get(criterion, {})
            if criterion_result.get('result', 'no').lower() == 'no':
                # 해당 기준 실패 시 피드백 생성
                feedback_prompt_path = os.path.join(self.prompt_dir, "SQL_feedback", feedback_file)
                try:
                    with open(feedback_prompt_path, 'r', encoding='utf-8') as f:
                        feedback_template = f.read().strip()
                    
                    # 프롬프트 포맷팅
                    prompt = feedback_template.replace(
                        "{question}", question
                    ).replace(
                        "{ss}", ss_content
                    ).replace(
                        "{scs}", ss_content  # SCS와 SS 둘 다 지원
                    ).replace(
                        "{generated_sql}", generated_sql
                    ).replace(
                        "{evaluation_result}", json.dumps(criterion_result, indent=2)
                    )
                    
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    
                    feedback_text = response.choices[0].message.content
                    feedbacks.append(f"{criterion} Feedback: {feedback_text}")
                    
                except Exception as e:
                    print(f"Error generating {criterion} feedback: {e}")
        
        return "\n\n".join(feedbacks) if feedbacks else ""
    
    def refine_sql_with_feedback(self, original_sql, feedback, question, ss_content, table_summary):
        """피드백을 바탕으로 SQL 재생성"""
        if not feedback:
            return original_sql
        
        refinement_prompt = f"""You are an expert at improving SQL queries based on feedback.

**Question:** {question}

**Table Summary:** {table_summary}

**SS (Structure Specification):** {ss_content}

**Original SQL:** {original_sql}

**Feedback for Improvement:** {feedback}

**Task:**
Based on the feedback provided, generate an improved SQL query that addresses all the issues mentioned. 
Focus on:
1. Using appropriate column roles as defined in the SS
2. Properly handling row analysis when needed
3. Using correct data types for operations
4. Maintaining faithfulness to the SS schema

Generate only the improved SQL query without any explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": refinement_prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            
            improved_sql = response.choices[0].message.content.strip()
            
            # SQL만 추출 (설명 제거)
            if "SELECT" in improved_sql.upper():
                # SQL 부분만 추출
                lines = improved_sql.split('\n')
                sql_lines = []
                for line in lines:
                    if any(keyword in line.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING']):
                        sql_lines.append(line.strip())
                    elif sql_lines and line.strip():  # SQL 시작 후 연속된 라인
                        sql_lines.append(line.strip())
                
                if sql_lines:
                    return ' '.join(sql_lines)
            
            return improved_sql
            
        except Exception as e:
            print(f"Error refining SQL with feedback: {e}")
            return original_sql
