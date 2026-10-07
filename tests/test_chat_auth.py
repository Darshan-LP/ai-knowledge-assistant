import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

TEST_USER = {
    "username": "test_user_pytest",
    "email": "pytest@example.com",
    "password": "TestPassword123!"
}


@pytest.fixture(scope="module")
def auth_token():
    # Register test user (ignore if already registered)
    client.post("/auth/register", json=TEST_USER)

    # Login using JSON payload matching schema
    response = client.post(
        "/auth/login",
        json={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        }
    )

    assert response.status_code == 200, f"Login failed: {response.text}"
    token = response.json().get("access_token")
    assert token is not None
    return token


def test_unauthenticated_chat():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 401


def test_authenticated_chat_flow(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Send message to chat endpoint
    response = client.post(
        "/chat",
        headers=headers,
        json={"message": "calculate 5 + 5"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert data["route"] == "tool"
    assert "10" in data["answer"]

    conversation_id = data["conversation_id"]

    # Fetch user conversations list
    conv_response = client.get("/chat/conversations", headers=headers)
    assert conv_response.status_code == 200
    conversations = conv_response.json()
    assert any(c["id"] == conversation_id for c in conversations)

    # Fetch specific conversation history
    hist_response = client.get(f"/chat/conversations/{conversation_id}", headers=headers)
    assert hist_response.status_code == 200
    history = hist_response.json()
    assert history["conversation_id"] == conversation_id
    assert len(history["messages"]) >= 2