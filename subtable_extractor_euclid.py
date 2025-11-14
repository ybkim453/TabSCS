import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

outliers=[]

class ColumnSimilarityAnalyzer:
    def __init__(self, model_name='BAAI/bge-large-en'):
        self.model = SentenceTransformer(model_name)
        
    def calculate_column_similarities(self, column_values):
        # 모든 값을 문자열로 변환 (NaN도 그대로 유지)
        str_values = [str(val) for val in column_values]
        
        if len(str_values) < 2:
            return {}
        
        # 임베딩 생성
        embeddings = self.model.encode(str_values)
        
        # 유클리드 거리 행렬 계산 후 유사도로 변환
        distance_matrix = euclidean_distances(embeddings)
        # 거리를 유사도로 변환 (거리가 작을수록 유사도가 높음)
        similarity_matrix = 1 / (1 + distance_matrix)
        
        # 각 값의 유사도 총합 및 평균 계산
        similarities = {}
        for i, val in enumerate(str_values):
            # 자기 자신을 제외한 다른 값들과의 유사도
            other_similarities = [similarity_matrix[i][j] for j in range(len(str_values)) if i != j]
            
            similarities[i] = {
                'value': val,
                'similarity_sum': sum(other_similarities),
                'similarity_avg': np.mean(other_similarities) if other_similarities else 0,
                'similarity_scores': other_similarities
            }
        
        return similarities
    
    def analyze_table_columns(self, table_df, log_file_path):
        column_analyses = {}
        
        for col in table_df.columns:
            log_message = f"  column '{col}' is being analyzed..."
            print(log_message)
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(log_message + '\n')
            
            similarities = self.calculate_column_similarities(table_df[col])
            
            if similarities:
                # 유사도 평균들의 분포 계산
                avg_similarities = [sim['similarity_avg'] for sim in similarities.values()]
                
                column_analyses[col] = {
                    'similarities': similarities,
                    'distribution_std': np.std(avg_similarities),
                    'min_similarity': min(avg_similarities),
                    'max_similarity': max(avg_similarities),
                    'mean_similarity': np.mean(avg_similarities)
                }
        
        return column_analyses
    
    def select_rows_for_subtable(self, table_df, column_analyses, log_file_path):
        selected_rows = []
        outliers = []
        
        # 특이값 행 3개 선택
        extreme_columns = sorted(column_analyses.items(), 
                               key=lambda x: x[1]['distribution_std'], 
                               reverse=True)
        
        anomaly_count = 0
        used_columns = set()
        
        for col_name, col_info in extreme_columns:
            if anomaly_count >= 3:  # Fix: 특이값 행 개수 변경 시 여기 수정 (예: 3개면 >= 3)
                break
                
            if col_name in used_columns:
                continue
                
            # 낮은 유사도를 가진 값의 행 찾기
            min_sim_row = None
            min_sim_value = float('inf')
            
            for row_idx, sim_info in col_info['similarities'].items():
                if sim_info['similarity_avg'] < min_sim_value:
                    min_sim_value = sim_info['similarity_avg']
                    min_sim_row = row_idx
            
            if min_sim_row is not None and min_sim_row not in selected_rows:
                selected_rows.append(min_sim_row)
                used_columns.add(col_name)
                anomaly_count += 1
                outliers.append(int(min_sim_row))
                log_message = f"  selected outlier row: row {min_sim_row} (column '{col_name}', similarity: {min_sim_value:.4f})"
                print(log_message)
                with open(log_file_path, 'a', encoding='utf-8') as log_f:
                    log_f.write(log_message + '\n')
        
        # 정상값 행 3개 선택
        row_scores = {}
        
        for row_idx in range(len(table_df)):
            if row_idx in selected_rows:
                continue
                
            total_score = 0
            valid_columns = 0
            
            for col_name, col_info in column_analyses.items():
                if row_idx in col_info['similarities']:
                    total_score += col_info['similarities'][row_idx]['similarity_avg']
                    valid_columns += 1
            
            if valid_columns > 0:
                row_scores[row_idx] = total_score / valid_columns
        
        # select the rows with the highest scores (exactly 3 rows)
        normal_rows = sorted(row_scores.items(), key=lambda x: x[1], reverse=True)
        normal_count = 0
        
        for row_idx, score in normal_rows:
            if normal_count >= 3:  # Fix: change the number of normal rows if needed (e.g., if 3 rows, >= 3)
                break
            if row_idx not in selected_rows:
                selected_rows.append(row_idx)
                log_message = f"  selected normal row: row {row_idx} (average similarity: {score:.4f})"
                print(log_message)
                with open(log_file_path, 'a', encoding='utf-8') as log_f:
                    log_f.write(log_message + '\n')
                normal_count += 1
        
        # sort the original table order and select exactly 6 rows
        selected_rows = sorted(list(set(selected_rows)))[:6]  # FIX: change the number of rows if needed (e.g., if 3X3, [:6])
        log_message = f"  final selected rows (original order): {selected_rows} (total {len(selected_rows)} rows)"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        return selected_rows, outliers
    
    # second JSON format (header + rows)
    def process_jsonl_record(self, jsonl_path, index, log_file_path):
        log_message = f"\nProcessing index {index} in JSONL file: {jsonl_path}"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        # load the specific index record from the JSONL file
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    record = json.loads(line.strip())
                    break
            else:
                raise IndexError(f"index {index} not found.")
        
        # extract table data (header + rows format)
        table = record['table']
        headers = table['header']
        data_rows = table['rows']
        
        log_message = f"  table size: {len(data_rows)} rows {len(headers)} columns"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        # create DataFrame (for analysis)
        table_df = pd.DataFrame(data_rows, columns=headers)
        
        # analyze column similarities
        column_analyses = self.analyze_table_columns(table_df, log_file_path)
        
        if not column_analyses:
            log_message = "  no columns available for analysis."
            print(log_message)
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(log_message + '\n')
            # select the top 6 data rows
            selected_rows = list(range(min(6, len(data_rows))))
            outliers = []
        else:
            # select rows for the subtable
            selected_rows, outliers = self.select_rows_for_subtable(table_df, column_analyses, log_file_path)
        
        # construct the subtable with the selected rows (header + selected data)
        result_lines = ['\t'.join(headers)]  # 헤더
        for i in selected_rows:
            result_lines.append('\t'.join(data_rows[i]))
        
        log_message = f"  subtable size: {len(result_lines)} lines"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        # return the result_lines, selected_rows, and outliers in the same format as process_single_table
        return result_lines, selected_rows, outliers

    def process_single_table(self, file_path, log_file_path):
        log_message = f"\nProcessing table: {file_path}"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        # read all lines from the original file (keep the original format)
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # header and separator (first 2 lines) - keep the original format (don't strip)
        header_line = all_lines[0].rstrip('\n\r')
        separator_line = all_lines[1].rstrip('\n\r')
        
        # data lines (from the 3rd line)
        data_lines = [line.strip() for line in all_lines[2:]]
        
        # use DataFrame only for analysis (use the first column as the index)
        temp_df = pd.read_csv(file_path, sep='\t', header=0, index_col=0, na_values=[''], keep_default_na=False)
        data_rows = temp_df.iloc[1:]  # data only (header is already the column names)
        
        # analyze column similarities
        column_analyses = self.analyze_table_columns(data_rows, log_file_path)
        
        if not column_analyses:
            log_message = "  no columns available for analysis."
            print(log_message)
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(log_message + '\n')
            # select the top 6 data lines
            selected_data_lines = data_lines[:6]  # FIX: change the number of rows if needed (e.g., if 3X3, [:6])
        else:
            # select rows for the subtable
            selected_rows, outliers = self.select_rows_for_subtable(
            data_rows, column_analyses, log_file_path
        )
            selected_data_lines = [data_lines[i] for i in selected_rows]
        
        # result: header + separator + selected 6 data lines
        result_lines = [header_line, separator_line] + selected_data_lines
        log_message = f"  subtable size: {len(result_lines)} lines"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        # return result_lines + selected indices
        return result_lines, selected_rows, outliers


def main():
    # JSONL processing example
    jsonl_path = "datasets/wtq.jsonl"  # change the JSONL file path
    index = 0  # index of the record to process
    output_dir = "subtables"
    
    os.makedirs(output_dir, exist_ok=True)
    analyzer = ColumnSimilarityAnalyzer()
    log_file_path = os.path.join(output_dir, "log.txt")
    
    with open(log_file_path, 'w', encoding='utf-8') as log_f:
        log_f.write("=== JSONL record processing started ===\n")

    try:
        result_lines, selected_rows = analyzer.process_jsonl_record(jsonl_path, index, log_file_path)
        
        # save the result
        output_path = os.path.join(output_dir, f"result_{index}.tsv")
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in result_lines:
                f.write(line + '\n')
        
        print(f"✅ saved: {output_path}")
        print(f"selected rows: {selected_rows}")

    except Exception as e:
        print(f"❌ error: {str(e)}")


if __name__ == "__main__":
    main()
