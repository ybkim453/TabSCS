import os
import re
import json
from openai import OpenAI

class TabSpecRowAnalysisEvaluator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        """Initialize TabSpec direct evaluator"""
        self.client = OpenAI()
        self.model_name = model_name
        self.prompt_dir = "prompt"
    
    def load_prompt(self, filename):
        """Load prompt file"""
        file_path = os.path.join(self.prompt_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Prompt file {filename} not found.")
            return ""
    
    def extract_row_analysis_from_tabspec(self, tabspec_content):
        """Extract Row Analysis from TabSpec"""
        # Find Row Analysis or Outlier Row Analysis section
        patterns = [
            r'\*\*Row Analysis:\*\*\n(.*?)(?=\n\*\*|\Z)',
            r'\*\*Outlier Row Analysis:\*\*\n(.*?)(?=\n\*\*|\Z)',
            r'Row Analysis:\n(.*?)(?=\n[A-Z]|\Z)',
            r'Outlier Row Analysis:\n(.*?)(?=\n[A-Z]|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, tabspec_content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_column_analysis_from_tabspec(self, tabspec_content):
        """Extract Column Analysis from TabSpec"""
        patterns = [
            r'\*\*Column Analysis:\*\*\n(.*?)(?=\n\*\*|\Z)',
            r'Column Analysis:\n(.*?)(?=\n[A-Z]|\Z)',
            r'Column Descriptions:\n(.*?)(?=\n[A-Z]|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, tabspec_content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def evaluate_tabspec_row_analysis(self, subtable_markdown, row_analysis_candidates, tabspec_content, log_file_path=None):
        """Evaluate TabSpec Row Analysis directly"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== TabSpec Row Analysis Direct Evaluation ===")
        
        # Extract Row Analysis from TabSpec
        row_analysis = self.extract_row_analysis_from_tabspec(tabspec_content)
        if not row_analysis:
            log("No row analysis found in TabSpec - treating as 'no special rows' case")
            return {
                'row_analysis_discrimination': {
                    'result': 'yes',
                    'reasoning': 'No row analysis provided - model determined no special rows exist. This is automatically considered correct.',
                    'issues': []
                },
                'accuracy_of_classification': {
                    'result': 'yes',
                    'reasoning': 'No row analysis identified, which is valid when no special rows exist.',
                    'issues': []
                },
                'overall_assessment': 'Model correctly determined no special rows exist.',
                'all_criteria_passed': True,
                'missing_row_analysis': ['None']
            }
        
        log(f"Extracted row analysis: {row_analysis}")
        
        # Load evaluation prompt
        evaluation_prompt_path = os.path.join(self.prompt_dir, "TabSpec_row_analysis_evaluation.txt")
        try:
            with open(evaluation_prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
        except FileNotFoundError:
            log(f"Failed to load evaluation prompt at {evaluation_prompt_path}")
            return None
        
        # Format prompt - match variable names in prompt template
        # 프롬프트에서 사용하는 변수명: evidence_sub_table, row_analysis, row_analysis_analysis
        prompt = prompt_template.replace(
            "{evidence_sub_table}", subtable_markdown
        ).replace(
            "{json.dumps(row_analysis, ensure_ascii=False)}", json.dumps(row_analysis_candidates, ensure_ascii=False)
        ).replace(
            "{row_analysis_analysis}", row_analysis
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at evaluating Row Analysis in TabSpec. Always return valid JSON following the exact format specified. Pay special attention to aggregate/summary rows and systematic patterns."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content
            log(f"LLM Response: {response_text}")
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                evaluation_result = json.loads(json_match.group())
                
                # Calculate all_criteria_passed (both criteria must pass)
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
    
    def evaluate_tabspec_column_analysis(self, subtable_markdown, tabspec_content, log_file_path=None):
        """Evaluate TabSpec Column Analysis directly (extendable)"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== TabSpec Column Analysis Direct Evaluation ===")
        
        # Extract Column Analysis from TabSpec
        column_analysis = self.extract_column_analysis_from_tabspec(tabspec_content)
        if not column_analysis:
            log("No column analysis found in TabSpec")
            return None
        
        log(f"Extracted column analysis: {column_analysis}")
        
        # Simple column analysis evaluation (extendable)
        # Currently only perform basic structure validation
        
        # Check column count
        table_lines = subtable_markdown.strip().split('\n')
        if len(table_lines) >= 2:
            header_line = table_lines[0]
            headers = [h.strip() for h in header_line.split('|')[1:-1]]
            
            # Check if each column is mentioned in analysis
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
    
    def evaluate_tabspec_comprehensive(self, subtable_markdown, row_analysis_candidates, tabspec_content, log_file_path=None):
        """Evaluate TabSpec comprehensive directly"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== TabSpec Comprehensive Direct Evaluation ===")
        
        # Evaluate Row Analysis
        row_evaluation = self.evaluate_tabspec_row_analysis(
            subtable_markdown, row_analysis_candidates, tabspec_content, log_file_path
        )
        
        # Evaluate Column Analysis
        column_evaluation = self.evaluate_tabspec_column_analysis(
            subtable_markdown, tabspec_content, log_file_path
        )
        
        # Overall result
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
