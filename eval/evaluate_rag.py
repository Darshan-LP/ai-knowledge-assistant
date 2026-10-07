import time
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

TEST_DATASET = [
    {
        "question": "What is the purpose of this AI assistant?",
        "expected_keywords": ["knowledge", "assistant", "information", "RAG"]
    },
    {
        "question": "How does authentication work in the system?",
        "expected_keywords": ["token", "JWT", "auth", "user", "login"]
    }
]

def get_auth_headers():
    # 1. Register or login a test user to retrieve JWT token
    user_data = {"username": "eval_user", "password": "evalpassword123"}
    client.post("/auth/register", json=user_data)
    
    login_res = client.post("/auth/login", data={"username": "eval_user", "password": "evalpassword123"})
    if login_res.status_code == 200:
        token = login_res.json().get("access_token")
        return {"Authorization": f"Bearer {token}"}
    return {}

def run_evaluation():
    print("==================================================")
    print("           RAG EVALUATION SUITE RUN              ")
    print("==================================================\n")
    
    headers = get_auth_headers()
    results = []
    
    for idx, item in enumerate(TEST_DATASET, 1):
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        
        start_time = time.time()
        
        # Authenticated POST request to chat endpoint
        response = client.post("/chat", json={"message": question}, headers=headers)
        elapsed_time = round(time.time() - start_time, 3)
        
        answer = ""
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", data.get("message", str(data)))
        else:
            answer = f"Status {response.status_code}: {response.text}"
        
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        relevance_score = round(len(found_keywords) / len(expected_keywords), 2) if expected_keywords else 1.0
        
        res = {
            "test_id": idx,
            "question": question,
            "status_code": response.status_code,
            "latency_seconds": elapsed_time,
            "relevance_score": relevance_score,
            "matched_keywords": found_keywords,
            "answer_preview": answer[:120] + "..." if len(answer) > 120 else answer
        }
        results.append(res)
        
        print(f"Test #{idx}: '{question}'")
        print(f"  Status: {response.status_code} | Latency: {elapsed_time}s | Relevance: {relevance_score * 100}%")
        print(f"  Answer: {res['answer_preview']}\n")
        
    print("==================================================")
    print("EVALUATION COMPLETE")
    print("==================================================")
    
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Saved detailed results to 'eval_results.json'")

if __name__ == "__main__":
    run_evaluation()