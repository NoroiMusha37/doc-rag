from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories import DBRepository


async def get_db_repo(
        session: AsyncSession = Depends(get_db)
) -> DBRepository:
    return DBRepository(session)


async def get_doc_parser(request: Request):
    return request.state.doc_parser
