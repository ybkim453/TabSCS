import os
import re
import json
import pandas as pd
from io import StringIO
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TabSpecReconstructionBasedEvaluator:
    def __init__(self, model_name="gpt-4o"):
        """Initialize TabSpec table evaluator"""
        self.client = OpenAI()
        self.model_name = model_name
        self.bert_model = SentenceTransformer('BAAI/bge-large-en')
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
    
    def step1_extract_schema(self, title, tabspec_content):
        """Step 1: Extract schema from TabSpec"""
        schema_prompt_path = os.path.join(self.prompt_dir, "Table_Generation", "Schema_Extraction.txt")
        try:
            with open(schema_prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Schema extraction prompt not found at {schema_prompt_path}")
            # Fallback prompt
            prompt_template = """You are an expert at extracting table schemas from TabSpec (Structure Specification) and titles.

***TASK***:
Given a table title and TabSpec, extract the table name and column headers to create the table structure.

***OUTPUT FORMAT***:
{
    "table_name": "<Table Title>",
    "column_headers": ["<Column 1>", "<Column 2>", ...],
    "estimated_rows": 6
}"""
        
        prompt = f"{prompt_template}\n\n***INPUT***:\nTitle: {title}\n\nTabSpec Content:\n{tabspec_content}\n\nExtract the schema:"
        
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
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return None
                
        except Exception as e:
            print(f"Error in schema extraction: {str(e)}")
            return None
    
    def step2_extract_instructions(self, tabspec_content):
        """Step 2: Extract generation instructions from TabSpec"""
        instruction_prompt_path = os.path.join(self.prompt_dir, "Table_Generation", "Instruction_Extraction.txt")
        try:
            with open(instruction_prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Instruction extraction prompt not found at {instruction_prompt_path}")
            # Fallback prompt
            prompt_template = """You are an expert at converting TabSpec (Structure Specification) into detailed table generation instructions.

***TASK***:
Given a TabSpec, convert descriptive information into specific instructions for table data generation.

***OUTPUT FORMAT***:
{
    "data_generation_instructions": ["Generate [column] following pattern [examples]"],
    "row_analysis_instructions": ["Include [special characteristic] for distinctive rows"],
    "column_relationship_rules": ["[Column A] should relate to [Column B] based on [role]"]
}"""
        
        prompt = f"{prompt_template}\n\n***TabSpec CONTENT***:\n{tabspec_content}\n\nExtract the instructions:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at converting TabSpec content into generation instructions. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return None
                
        except Exception as e:
            print(f"Error in instruction extraction: {str(e)}")
            return None
    
    def step3_generate_table(self, schema, instructions):
        """Step 3: Generate table with schema and instructions"""
        generation_prompt_path = os.path.join(self.prompt_dir, "Table_Generation", "Table_Generation.txt")
        try:
            with open(generation_prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Table generation prompt not found at {generation_prompt_path}")
            # Fallback prompt
            prompt_template = """You are an expert at generating realistic tabular data.

***TASK***:
Generate a complete table using the provided schema and instructions.

***OUTPUT FORMAT***:
Generate EXACTLY 6 data rows following the schema and instructions.
Use "|" to separate cells and maintain proper markdown table formatting."""
        
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
        """Convert markdown table to DataFrame"""
        lines = markdown_text.strip().split('\n')
        
        # Find table part
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
        
        # Extract headers
        header_line = table_lines[0]
        headers = [h.strip() for h in header_line.split('|')[1:-1]]
        
        # Extract data rows (excluding separator lines)
        data_rows = []
        for line in table_lines[2:]:  # Exclude header and separator lines
            if line.strip() and '|' in line:
                row = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(row) == len(headers):
                    data_rows.append(row)
        
        if not data_rows:
            return None
        
        return pd.DataFrame(data_rows, columns=headers)
    
    def calculate_header_em(self, original_headers, generated_headers):
        """Calculate Header Exact Match score"""
        if not original_headers or not generated_headers:
            return 0.0
        
        # Normalize (lowercase, remove whitespace)
        orig_normalized = [h.lower().strip() for h in original_headers]
        gen_normalized = [h.lower().strip() for h in generated_headers]
        
        # Check for exact match
        if orig_normalized == gen_normalized:
            return 1.0
        
        # Calculate partial match score
        matches = sum(1 for h in orig_normalized if h in gen_normalized)
        return matches / max(len(orig_normalized), len(gen_normalized))
    
    def calculate_cell_similarity(self, original_df, generated_df):
        """Calculate Cell Similarity (BERT-based) score"""
        if original_df is None or generated_df is None:
            return 0.0
        
        # Compare only common columns
        common_cols = list(set(original_df.columns) & set(generated_df.columns))
        if not common_cols:
            return 0.0
        
        similarities = []
        
        for col in common_cols:
            orig_values = original_df[col].astype(str).tolist()
            gen_values = generated_df[col].astype(str).tolist()
            
            # Align to minimum length
            min_len = min(len(orig_values), len(gen_values))
            orig_values = orig_values[:min_len]
            gen_values = gen_values[:min_len]
            
            if orig_values and gen_values:
                # BERT embedding
                orig_embeddings = self.bert_model.encode(orig_values)
                gen_embeddings = self.bert_model.encode(gen_values)
                
                # Calculate cosine similarity
                col_similarities = []
                for i in range(len(orig_values)):
                    sim = cosine_similarity([orig_embeddings[i]], [gen_embeddings[i]])[0][0]
                    col_similarities.append(sim)
                
                similarities.extend(col_similarities)
        
        return np.mean(similarities) if similarities else 0.0
    
    def evaluate_tabspec_by_table_generation(self, title, tabspec_content, subtable_df, log_file_path=None):
        """Evaluate TabSpec by table generation"""
        
        def log(msg):
            print(msg)
            if log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        
        log("=== TabSpec Table Generation Evaluation ===")
        
        # Step 1: Extract schema
        log("Step 1: Extracting schema...")
        schema = self.step1_extract_schema(title, tabspec_content)
        if not schema:
            log("Failed to extract schema")
            return None
        
        log(f"Extracted schema: {schema}")
        
        # Step 2: Extract instructions
        log("Step 2: Extracting instructions...")
        instructions = self.step2_extract_instructions(tabspec_content)
        if not instructions:
            log("Failed to extract instructions")
            return None
        
        log(f"Extracted instructions: {instructions}")
        
        # Step 3: Generate table
        log("Step 3: Generating table...")
        generated_table_text = self.step3_generate_table(schema, instructions)
        if not generated_table_text:
            log("Failed to generate table")
            return None
        
        log(f"Generated table text:\n{generated_table_text}")
        
        # Parse generated table
        generated_df = self.parse_markdown_table(generated_table_text)
        if generated_df is None:
            log("Failed to parse generated table")
            return None
        
        log(f"Parsed generated table: {generated_df.shape}")
        log(f"Original subtable columns: {subtable_df.columns.tolist()}")
        log(f"Generated table columns: {generated_df.columns.tolist()}")
        
        # Calculate evaluation metrics
        original_headers = subtable_df.columns.tolist()
        generated_headers = generated_df.columns.tolist()
        
        header_em = self.calculate_header_em(original_headers, generated_headers)
        cell_similarity = self.calculate_cell_similarity(subtable_df, generated_df)
        
        log(f"Header EM Score: {header_em:.4f}")
        log(f"Cell Similarity: {cell_similarity:.4f}")
        
        # Return result
        return {
            'header_em_score': header_em,
            'cell_similarity_score': cell_similarity,
            'generated_table': generated_table_text,
            'generated_df': generated_df,
            'schema': schema,
            'instructions': instructions,
            'evaluation_passed': header_em >= 1.0 and cell_similarity >= 0.85  # Threshold
        }
