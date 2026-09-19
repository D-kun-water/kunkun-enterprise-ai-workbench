"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.core import get_runtime, get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_runtime().bootstrap()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="面向普通员工的企业制度检索、问答与基础评测 API",
    lifespan=lifespan,
)
app.include_router(router)
