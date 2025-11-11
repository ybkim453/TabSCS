"""
SS Row Analysis Evaluator - 직접 LLM 평가를 통한 SS 평가 모듈
TabBridge에서 사용할 SS 평가 시스템 (방식 1: 직접 평가)
"""

import os
import re
import json
from openai import OpenAI

class SSRowAnalysisEvaluator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        """SS 직접 평가기 초기화"""
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
    
    def extract_row_analysis_from_ss(self, ss_content):
        """SS에서 Row Analysis 부분 추출"""
        # Row Analysis 또는 Outlier Row Analysis 섹션 찾기
        patterns = [
            r'\*\*Row Analysis:\*\*\n(.*?)(?=\n\*\*|\Z)',
            r'\*\*Outlier Row Analysis:\*\*\n(.*?)(?=\n\*\*|\Z)',
            r'Row Analysis:\n(.*?)(?=\n[A-Z]|\Z)',
            r'Outlier Row Analysis:\n(.*?)(?=\n[A-Z]|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ss_content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_column_analysis_from_ss(self, ss_content):
        """SS에서 Column Analysis 부분 추출"""
        patterns = [
            r'\*\*Column Analysis:\*\*\n(.*?)(?=\n\*\*|\Z)',
            r'Column Analysis:\n(.*?)(?=\n[A-Z]|\Z)',
            r'Column Descriptions:\n(.*?)(?=\n[A-Z]|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ss_content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def evaluate_ss_row_analysis(self, subtable_markdown, outlier_candidates, ss_content, log_file_path=None):
        """SS의 Row Analysis를 직접 평가"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== SS Row Analysis Direct Evaluation ===")
        
        # SS에서 Row Analysis 추출
        row_analysis = self.extract_row_analysis_from_ss(ss_content)
        if not row_analysis:
            log("No row analysis found in SS - treating as 'no outliers' case")
            return {
                'row_analysis_discrimination': {
                    'result': 'yes',
                    'reasoning': 'No row analysis provided - model determined no outliers exist. This is automatically considered correct.',
                    'issues': []
                },
                'accuracy_of_classification': {
                    'result': 'yes',
                    'reasoning': 'No outliers identified, which is valid when no special rows exist.',
                    'issues': []
                },
                'overall_assessment': 'Model correctly determined no outliers exist.',
                'all_criteria_passed': True,
                'missing_row_analysis': ['None']
            }
        
        log(f"Extracted row analysis: {row_analysis}")
        
        # 평가 프롬프트 로드
        evaluation_prompt_path = os.path.join(self.prompt_dir, "SS_row_analysis_evaluation.txt")
        try:
            with open(evaluation_prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
        except FileNotFoundError:
            log(f"Failed to load evaluation prompt at {evaluation_prompt_path}")
            return None
        
        # 프롬프트 포맷팅 - 프롬프트 템플릿의 변수명과 일치시킴
        # 프롬프트에서 사용하는 변수명: evidence_sub_table, row_analysis, row_analysis_analysis
        prompt = prompt_template.replace(
            "{evidence_sub_table}", subtable_markdown
        ).replace(
            "{json.dumps(row_analysis, ensure_ascii=False)}", json.dumps(outlier_candidates, ensure_ascii=False)
        ).replace(
            "{row_analysis_analysis}", row_analysis
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at evaluating Row Analysis in SS. Always return valid JSON following the exact format specified. Pay special attention to aggregate/summary rows and systematic patterns."
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
                
                # all_criteria_passed 계산 (두 기준 모두 통과해야 함)
                row_discrimination = evaluation_result.get('row_analysis_discrimination', {}).get('result', 'no')
                accuracy_classification = evaluation_result.get('accuracy_of_classification', {}).get('result', 'no')
                
                evaluation_result['all_criteria_passed'] = (
                    row_discrimination.lower() == 'yes' and 
                    accuracy_classification.lower() == 'yes'
                )
                
                log(f"Row Analysis Discrimination: {row_discrimination}")
                log(f"Accuracy of Classification: {accuracy_classification}")
                log(f"All Criteria Passed: {evaluation_result['all_criteria_passed']}")
                
                return evaluation_result
            else:
                log("No JSON found in evaluation response")
                return None
                
        except Exception as e:
            log(f"Error in row analysis evaluation: {str(e)}")
            return None
    
    def evaluate_ss_column_analysis(self, subtable_markdown, ss_content, log_file_path=None):
        """SS의 Column Analysis를 직접 평가 (확장 가능)"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== SS Column Analysis Direct Evaluation ===")
        
        # SS에서 Column Analysis 추출
        column_analysis = self.extract_column_analysis_from_ss(ss_content)
        if not column_analysis:
            log("No column analysis found in SS")
            return None
        
        log(f"Extracted column analysis: {column_analysis}")
        
        # 간단한 컬럼 분석 평가 (확장 가능)
        # 현재는 기본적인 구조 검증만 수행
        
        # 컬럼 개수 확인
        table_lines = subtable_markdown.strip().split('\n')
        if len(table_lines) >= 2:
            header_line = table_lines[0]
            headers = [h.strip() for h in header_line.split('|')[1:-1]]
            
            # 각 컬럼이 분석에 언급되었는지 확인
            missing_columns = []
            for header in headers:
                if header.lower() not in column_analysis.lower():
                    missing_columns.append(header)
            
            return {
                'column_coverage': len(headers) - len(missing_columns),
                'total_columns': len(headers),
                'missing_columns': missing_columns,
                'coverage_ratio': (len(headers) - len(missing_columns)) / len(headers) if headers else 0,
                'evaluation_passed': len(missing_columns) == 0
            }
        
        return None
    
    def evaluate_ss_comprehensive(self, subtable_markdown, outlier_candidates, ss_content, log_file_path=None):
        """SS 종합 직접 평가"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== SS Comprehensive Direct Evaluation ===")
        
        # Row Analysis 평가
        row_evaluation = self.evaluate_ss_row_analysis(
            subtable_markdown, outlier_candidates, ss_content, log_file_path
        )
        
        # Column Analysis 평가
        column_evaluation = self.evaluate_ss_column_analysis(
            subtable_markdown, ss_content, log_file_path
        )
        
        # 종합 결과
        overall_passed = False
        if row_evaluation and column_evaluation:
            overall_passed = (
                row_evaluation.get('all_criteria_passed', False) and
                column_evaluation.get('evaluation_passed', False)
            )
        elif row_evaluation:
            overall_passed = row_evaluation.get('all_criteria_passed', False)
        
        log(f"Overall evaluation passed: {overall_passed}")
        
        return {
            'row_analysis_evaluation': row_evaluation,
            'column_analysis_evaluation': column_evaluation,
            'all_criteria_passed': overall_passed,
            'evaluation_method': 'direct_llm_evaluation'
        }
