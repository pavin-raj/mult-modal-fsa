"""
Production-grade Tenant Context extraction (FastAPI Depends pattern).

This is the canonical way to get "who this request belongs to" in a real SaaS.

Current (prototype):
- Exclusively reads from headers X-Tenant-ID and X-Industry.
- Returns strongly typed TenantContext.
- Raises 401 if tenant identity is missing (strict production behavior).

Future (when you add real auth):
- Replace the extraction logic with JWT validation (Auth0, Supabase, Firebase, Clerk, or self-signed).
- Validate exp, signature, and claims (tenant_id, industry, features, licensed_industries).
- Optionally hydrate the full TenantContext from your tenant service / DB using the tenant_id from the JWT.

Usage:
    from fastapi import Depends
    from backend.dependencies.tenant import get_current_tenant

    @app.post("/get-guidance")
    async def get_guidance(tenant: TenantContext = Depends(get_current_tenant)):
        ...
"""

from typing import Optional
from fastapi import Header, HTTPException
import structlog

from ai_core.models.schemas import Industry, TenantContext

logger = structlog.get_logger(__name__)


async def get_current_tenant(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_industry: Optional[str] = Header(None, alias="X-Industry"),
) -> TenantContext:
    """
    Production-style tenant resolver.

    Headers are the source of truth. This is what field apps, web frontends,
    and hardware devices should send.
    """

    if not x_tenant_id:
        logger.warning("auth.missing_tenant_header")
        raise HTTPException(
            status_code=401,
            detail="Missing X-Tenant-ID header. This is a multi-tenant system."
        )

    # Normalize industry (default to construction if not provided or invalid)
    try:
        industry = Industry(x_industry.lower()) if x_industry else Industry.CONSTRUCTION
    except ValueError:
        industry = Industry.CONSTRUCTION

    tenant = TenantContext(
        tenant_id=str(x_tenant_id),
        industry=industry,
        # In production these would come from the JWT claims or a tenant lookup service
        licensed_industries=[industry],
        company_name=f"{industry.value.replace('_', ' ').title()} Customer",
        features=["rag", "vision", "director", "basic_safety", "document_upload"],
        is_active=True,
    )

    logger.info(
        "tenant.resolved",
        tenant_id=tenant.tenant_id,
        industry=tenant.industry.value
    )

    return tenant