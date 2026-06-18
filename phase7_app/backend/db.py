import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from .models import Base

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Determine database URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback to SQLite for development if no Postgres is provided
    logger.warning("DATABASE_URL not found. Falling back to SQLite.")
    # SQLite async requires aiosqlite
    DATABASE_URL = "sqlite+aiosqlite:///./code_review.db"
elif DATABASE_URL.startswith("postgres://"):
    # Fix for SQLAlchemy 1.4+ which requires postgresql://
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    # Ensure asyncpg is used
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

logger.info(f"Using database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")

# Create engine
engine = create_async_engine(
    DATABASE_URL, 
    echo=os.getenv("DEBUG", "False").lower() == "true",
    future=True
)

# Create session factory
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def init_db():
    """Initializes the database tables."""
    try:
        async with engine.begin() as conn:
            # await conn.run_sync(Base.metadata.drop_all) # Be careful with this in production!
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

async def get_db():
    """Dependency for FastAPI to get DB session."""
    session = async_session()
    try:
        yield session
    finally:
        await session.close()
