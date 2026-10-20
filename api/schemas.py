from pydantic import BaseModel, Field
from typing import Optional


class LoanApplication(BaseModel):
    revolving_utilization: float = Field(
        ..., example=0.5,
        description="Revolving credit utilization rate (0–1)"
    )
    age: int = Field(..., example=45, description="Age of borrower in years (>= 1)")
    past_due_30_59_days: int = Field(
        ..., example=0,
        description="# times 30–59 days past due in last 2 years"
    )
    debt_ratio: float = Field(
        ..., example=0.3,
        description="Monthly debt payments / gross monthly income"
    )
    monthly_income: Optional[float] = Field(
        None, example=5000.0,
        description="Monthly gross income in dollars (null → treated as 0)"
    )
    open_credit_lines: int = Field(
        ..., example=8,
        description="# open loans and lines of credit"
    )
    times_90_days_late: int = Field(
        ..., example=0,
        description="# times 90+ days past due"
    )
    real_estate_loans: int = Field(
        ..., example=1,
        description="# mortgage and real estate loans or lines"
    )
    past_due_60_89_days: int = Field(
        ..., example=0,
        description="# times 60–89 days past due in last 2 years"
    )
    num_dependents: Optional[int] = Field(
        None, example=2,
        description="# dependents (not including self; null → treated as 0)"
    )


class PredictionResponse(BaseModel):
    default_probability: float
    prediction: int   # 0 = no default, 1 = default
    risk_level: str   # low / medium / high
