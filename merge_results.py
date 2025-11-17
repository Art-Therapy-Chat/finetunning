import json
import csv

# 1. LoRa 결과 읽기
with open('layer_freezing/test_results/lora_test_results_20251117_144649.json', 'r', encoding='utf-8') as f:
    lora_data = json.load(f)

# 2. Base Model 결과 읽기
with open('test_results/base_model_test_results_20251117_145654.json', 'r', encoding='utf-8') as f:
    base_data = json.load(f)

# 3. LoRa htp_interpretations.json 읽기 (원본 학습 데이터)
with open('LoRa/htp_interpretations.json', 'r', encoding='utf-8') as f:
    original_lora = json.load(f)

# 데이터를 input 기준으로 매핑
data_map = {}

# LoRa 테스트 결과 추가
for item in lora_data['test_cases']:
    input_text = item['input']
    if input_text not in data_map:
        data_map[input_text] = {}
    data_map[input_text]['lora'] = item['model_output']

# Base Model 결과 추가
for item in base_data['test_cases']:
    input_text = item['input']
    if input_text not in data_map:
        data_map[input_text] = {}
    data_map[input_text]['base_model'] = item['model_output']

# 원본 LoRa interpretation 추가 (있는 경우)
for item in original_lora:
    input_text = item['input']
    if input_text in data_map:
        data_map[input_text]['original_lora'] = item['interpretation']

# CSV 생성
output_file = 'model_comparison_results.csv'
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    
    # 헤더
    writer.writerow(['input', 'LoRa_finetune', 'Base_Model', 'Original_LoRa_Interpretation'])
    
    # 데이터 작성
    for input_text, outputs in data_map.items():
        writer.writerow([
            input_text,
            outputs.get('lora', ''),
            outputs.get('base_model', ''),
            outputs.get('original_lora', '')
        ])

print(f"✅ CSV 파일 생성 완료: {output_file}")
print(f"   총 {len(data_map)}개 항목")
print(f"   - LoRa 결과: {sum(1 for v in data_map.values() if 'lora' in v)}개")
print(f"   - Base Model 결과: {sum(1 for v in data_map.values() if 'base_model' in v)}개")
print(f"   - 원본 LoRa Interpretation: {sum(1 for v in data_map.values() if 'original_lora' in v)}개")
