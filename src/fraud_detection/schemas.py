from pydantic import BaseModel, Field


class TransactionData(BaseModel):

    sender_account: str
    receiver_account: str

    sender_bank: str
    receiver_bank: str

    transaction_type: str

    amount: float = Field(
        gt=0
    )

    transaction_hour: int

    medium: str

    transaction_date: int


class FraudDetectionRequest(BaseModel):

    transaction_id: str

    transaction: TransactionData

    history: list[TransactionData] = []


class FraudDetectionResponse(BaseModel):

    transaction_id: str

    anomaly_score: float

    threshold: float

    is_fraud: bool

    risk_level: str

    model: str