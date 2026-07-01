import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA_DATASET_PATH = ROOT / "data" / "evaluate_data" / "qa_dataset_50_questions.json"
CHATBOT_ANSWERS_PATH = ROOT / "tests" / "results" / "my_chatbot_answers.json"

def main():
    if not QA_DATASET_PATH.exists():
        print(f"Error: Could not find file {QA_DATASET_PATH}")
        return
    if not CHATBOT_ANSWERS_PATH.exists():
        print(f"Error: Could not find file {CHATBOT_ANSWERS_PATH}")
        return

    # Load chatbot answers and map query -> answer
    print("Reading chatbot answers...")
    with open(CHATBOT_ANSWERS_PATH, "r", encoding="utf-8") as f:
        answers_data = json.load(f)
    
    query_to_answer = {}
    for item in answers_data:
        query = item.get("query", "").strip()
        answer = item.get("answer", "").strip()
        if query and answer:
            query_to_answer[query] = answer

    # Load 50 questions dataset
    print("Reading 50 questions...")
    with open(QA_DATASET_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Merge answers
    updated_count = 0
    for item in qa_data:
        question = item.get("question", "").strip()
        if question in query_to_answer:
            item["chatbot_answer"] = query_to_answer[question]
            updated_count += 1
        else:
            # Try fuzzy match if exact match fails due to minor whitespaces
            matched = False
            for q, a in query_to_answer.items():
                if q.replace(" ", "") == question.replace(" ", ""):
                    item["chatbot_answer"] = a
                    updated_count += 1
                    matched = True
                    break
            if not matched:
                item["chatbot_answer"] = ""

    # Save back to qa_dataset_50_questions.json
    with open(QA_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)

    print(f"Success! Updated {updated_count}/{len(qa_data)} answers to dataset file.")

if __name__ == "__main__":
    main()
