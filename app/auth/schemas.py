from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=50
    )

    email: str = Field(
        min_length=5,
        max_length=100
    )

    password: str = Field(
        min_length=8,
        max_length=100
    )


class LoginRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=50
    )

    password: str = Field(
        min_length=8,
        max_length=100
    )


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"