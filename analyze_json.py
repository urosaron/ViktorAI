import json

# Load the JSON file
with open('benchmark_results/gemma3/gemma3:1b/run_20250314_173508/raw_data/benchmark_results_gemma3:1b_20250314_173508.json') as f:
    data = json.load(f)

# Print basic information
print("Number of questions:", len(data['metadata']['questions']))
print("Number of prompt categories:", len(data['metadata']['prompt_categories']))
print("\nResponses per category:")
for category in data['metadata']['prompt_categories']:
    print(f"  {category}: {len(data['responses'].get(category, []))}")

print("\nMetrics per category:")
for category in data['metadata']['prompt_categories']:
    print(f"  {category}: {len(data['metrics'].get(category, []))}")

# Count total responses and evaluations
total_responses = sum(len(data['responses'].get(category, [])) for category in data['metadata']['prompt_categories'])
total_metrics = sum(len(data['metrics'].get(category, [])) for category in data['metadata']['prompt_categories'])

print(f"\nTotal responses: {total_responses}")
print(f"Total evaluations: {total_metrics}")

# Check if the number of questions matches the number of responses per category
num_questions = len(data['metadata']['questions'])
for category in data['metadata']['prompt_categories']:
    num_responses = len(data['responses'].get(category, []))
    if num_responses != num_questions:
        print(f"\nWarning: {category} has {num_responses} responses but there are {num_questions} questions") 