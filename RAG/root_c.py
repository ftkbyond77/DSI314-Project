import os

root_path = r"C:\Users\BM MONEY\coding\part2\student_assistant\RAG"

for root, dirs, files in os.walk(root_path):
    print(f"📂 {root}")
    for file in files:
        print(f"   └── {file}")