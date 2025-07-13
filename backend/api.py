from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.main import rank_products

app = FastAPI(title="Walmart Product Ranker API")

class PurchaseRecord(BaseModel):
    product_id: str = Field(..., description="ID of the purchased product")
    customer_zip: str = Field(
        ...,
        pattern=r"^\d{6}$",
        description="6-digit Indian postal code"
    )
    satisfaction_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Customer satisfaction score between 0.0 and 1.0"
    )

class RankRequest(BaseModel):
    # Now accepts string IDs
    product_ids: List[str] = Field(..., description="Product IDs to rank")
    customer_zip: str = Field(
        ...,
        pattern=r"^\d{6}$",
        description="6-digit Indian postal code"
    )
    new_purchase: Optional[PurchaseRecord] = Field(
        None,
        description="Optional: new purchase data for online model update"
    )

@app.post("/rank-products")
async def get_ranked_products(request: RankRequest):
    """
    Rank products by predicted relevance score (distance + rating).
    """
    try:
        purchase_data = request.new_purchase.dict() if request.new_purchase else None
        ranked = rank_products(
            request.product_ids,
            request.customer_zip,
            purchase_data
        )
        return {"ranked_products": ranked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
