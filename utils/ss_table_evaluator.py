"""
SS Table Evaluator - 테이블 생성 후 비교를 통한 SS 평가 모듈
TabBridge에서 사용할 SS 평가 시스템 (방식 2: 간접 평가)
"""

import os
import re
import json
import pandas as pd
from io import StringIO
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SSTableEvaluator:
    def __init__(self, model_name="gpt-4o"):
        """SS 테이블 평가기 초기화"""
        self.client = OpenAI()
        self.model_name = model_name
        self.bert_model = SentenceTransformer('BAAI/bge-large-en')
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
    
    def step1_extract_schema(self, title, ss_content):
        """Step 1: SS에서 스키마 추출"""
        prompt_template = self.load_prompt("Table_Generation/Schema_Extraction.txt")
        if not prompt_template:
            # Fallback 프롬프트
            prompt_template = """Extract table schema from the given information.
            
***TASK***:
Extract the table name and column headers from the title and SS content.

***OUTPUT FORMAT***:
{
    "table_name": "<table name>",
    "headers": ["<header1>", "<header2>", ...]
}"""
        
        prompt = f"{prompt_template}\n\n***INPUT***:\nTitle: {title}\n\nSS Content:\n{ss_content}\n\nExtract the schema:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting table schemas. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # JSON 추출
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return None
                
        except Exception as e:
            print(f"Error in schema extraction: {str(e)}")
            return None
    
    def step2_extract_instructions(self, ss_content):
        """Step 2: SS에서 생성 지침 추출"""
        prompt_template = self.load_prompt("Table_Generation/Instruction_Extraction.txt")
        if not prompt_template:
            # Fallback 프롬프트
            prompt_template = """Extract detailed instructions for table generation from SS content.
            
***TASK***:
Convert the SS content into detailed instructions for generating realistic table data.

***OUTPUT FORMAT***:
{
    "column_instructions": {
        "<column1>": "<generation instruction>",
        "<column2>": "<generation instruction>"
    },
    "row_count": <number>,
    "special_requirements": ["<requirement1>", "<requirement2>"]
}"""
        
        prompt = f"{prompt_template}\n\n***SS CONTENT***:\n{ss_content}\n\nExtract the instructions:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at converting SS content into generation instructions. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            
            # JSON 추출
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return None
                
        except Exception as e:
            print(f"Error in instruction extraction: {str(e)}")
            return None
    
    def step3_generate_table(self, schema, instructions):
        """Step 3: 스키마와 지침으로 테이블 생성"""
        prompt_template = self.load_prompt("Table_Generation/Table_Generation.txt")
        if not prompt_template:
            # Fallback 프롬프트
            prompt_template = """Generate a realistic table based on the given schema and instructions.
            
***TASK***:
Create a complete table with realistic data following the schema and instructions.

***OUTPUT FORMAT***:
Return the table in markdown format with proper headers and data rows."""
        
        prompt = f"{prompt_template}\n\n***SCHEMA***:\n{json.dumps(schema, indent=2)}\n\n***INSTRUCTIONS***:\n{json.dumps(instructions, indent=2)}\n\nGenerate the table:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert table generator. Generate realistic, coherent tables."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in table generation: {str(e)}")
            return None
    
    def parse_markdown_table(self, markdown_text):
        """마크다운 테이블을 DataFrame으로 변환"""
        lines = markdown_text.strip().split('\n')
        
        # 테이블 부분 찾기
        table_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line and not in_table:
                in_table = True
            if in_table and '|' in line:
                table_lines.append(line.strip())
            elif in_table and '|' not in line:
                break
        
        if len(table_lines) < 2:
            return None
        
        # 헤더 추출
        header_line = table_lines[0]
        headers = [h.strip() for h in header_line.split('|')[1:-1]]
        
        # 데이터 행 추출 (구분선 제외)
        data_rows = []
        for line in table_lines[2:]:  # 헤더와 구분선 제외
            if line.strip() and '|' in line:
                row = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(row) == len(headers):
                    data_rows.append(row)
        
        if not data_rows:
            return None
        
        return pd.DataFrame(data_rows, columns=headers)
    
    def calculate_header_em(self, original_headers, generated_headers):
        """Header Exact Match 점수 계산"""
        if not original_headers or not generated_headers:
            return 0.0
        
        # 정규화 (소문자, 공백 제거)
        orig_normalized = [h.lower().strip() for h in original_headers]
        gen_normalized = [h.lower().strip() for h in generated_headers]
        
        # 완전 일치 확인
        if orig_normalized == gen_normalized:
            return 1.0
        
        # 부분 일치 점수
        matches = sum(1 for h in orig_normalized if h in gen_normalized)
        return matches / max(len(orig_normalized), len(gen_normalized))
    
    def calculate_cell_similarity(self, original_df, generated_df):
        """Cell Similarity (BERT 기반) 점수 계산"""
        if original_df is None or generated_df is None:
            return 0.0
        
        # 공통 컬럼만 비교
        common_cols = list(set(original_df.columns) & set(generated_df.columns))
        if not common_cols:
            return 0.0
        
        similarities = []
        
        for col in common_cols:
            orig_values = original_df[col].astype(str).tolist()
            gen_values = generated_df[col].astype(str).tolist()
            
            # 최소 길이로 맞춤
            min_len = min(len(orig_values), len(gen_values))
            orig_values = orig_values[:min_len]
            gen_values = gen_values[:min_len]
            
            if orig_values and gen_values:
                # BERT 임베딩
                orig_embeddings = self.bert_model.encode(orig_values)
                gen_embeddings = self.bert_model.encode(gen_values)
                
                # 코사인 유사도 계산
                col_similarities = []
                for i in range(len(orig_values)):
                    sim = cosine_similarity([orig_embeddings[i]], [gen_embeddings[i]])[0][0]
                    col_similarities.append(sim)
                
                similarities.extend(col_similarities)
        
        return np.mean(similarities) if similarities else 0.0
    
    def evaluate_ss_by_table_generation(self, title, ss_content, subtable_df, log_file_path=None):
        """SS를 테이블 생성을 통해 평가"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== SS Table Generation Evaluation ===")
        
        # Step 1: 스키마 추출
        log("Step 1: Extracting schema...")
        schema = self.step1_extract_schema(title, ss_content)
        if not schema:
            log("Failed to extract schema")
            return None
        
        log(f"Extracted schema: {schema}")
        
        # Step 2: 지침 추출
        log("Step 2: Extracting instructions...")
        instructions = self.step2_extract_instructions(ss_content)
        if not instructions:
            log("Failed to extract instructions")
            return None
        
        log(f"Extracted instructions: {instructions}")
        
        # Step 3: 테이블 생성
        log("Step 3: Generating table...")
        generated_table_text = self.step3_generate_table(schema, instructions)
        if not generated_table_text:
            log("Failed to generate table")
            return None
        
        log(f"Generated table text:\n{generated_table_text}")
        
        # 생성된 테이블 파싱
        generated_df = self.parse_markdown_table(generated_table_text)
        if generated_df is None:
            log("Failed to parse generated table")
            return None
        
        log(f"Parsed generated table: {generated_df.shape}")
        
        # 평가 지표 계산
        header_em = self.calculate_header_em(subtable_df.columns.tolist(), generated_df.columns.tolist())
        cell_similarity = self.calculate_cell_similarity(subtable_df, generated_df)
        
        log(f"Header EM Score: {header_em:.4f}")
        log(f"Cell Similarity: {cell_similarity:.4f}")
        
        # 결과 반환
        return {
            'header_em_score': header_em,
            'cell_similarity_score': cell_similarity,
            'generated_table': generated_table_text,
            'generated_df': generated_df,
            'schema': schema,
            'instructions': instructions,
            'evaluation_passed': header_em >= 0.8 and cell_similarity >= 0.65  # 임계값
        }
