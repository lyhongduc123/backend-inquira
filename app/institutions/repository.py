"""
Repository for institution database operations.
Handles CRUD operations for institutions.
"""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.institutions import DBInstitution
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class InstitutionRepository:
    """Repository for institution database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_institution_by_id(self, institution_id: str) -> Optional[DBInstitution]:
        """Get institution by OpenAlex institution ID"""
        result = await self.db.execute(
            select(DBInstitution).where(DBInstitution.institution_id == institution_id)
        )
        return result.scalar_one_or_none()
    
    async def get_institution_by_ror(self, ror_id: str) -> Optional[DBInstitution]:
        """Get institution by ROR ID"""
        result = await self.db.execute(
            select(DBInstitution).where(DBInstitution.ror_id == ror_id)
        )
        return result.scalar_one_or_none()
    
    async def create_institution(self, institution_data: dict) -> DBInstitution:
        """
        Create a new institution record.
        
        Args:
            institution_data: Dictionary containing institution fields
            
        Returns:
            Created DBInstitution object
        """
        db_institution = DBInstitution(**institution_data)
        self.db.add(db_institution)
        await self.db.commit()
        await self.db.refresh(db_institution)
        
        logger.info(f"Created institution {institution_data.get('name')} ({institution_data.get('institution_id')})")
        return db_institution
    
    async def update_institution(self, institution_id: str, institution_data: dict) -> Optional[DBInstitution]:
        """
        Update existing institution record.
        
        Args:
            institution_id: OpenAlex institution ID
            institution_data: Dictionary containing updated fields
            
        Returns:
            Updated DBInstitution object or None if not found
        """
        institution = await self.get_institution_by_id(institution_id)
        if not institution:
            return None
        
        for key, value in institution_data.items():
            if hasattr(institution, key):
                setattr(institution, key, value)
        
        await self.db.commit()
        await self.db.refresh(institution)
        
        logger.info(f"Updated institution {institution_id}")
        return institution
    
    async def upsert_institution(self, institution_data: dict) -> DBInstitution:
        """
        Create or update institution record.
        
        Args:
            institution_data: Dictionary containing institution fields
            
        Returns:
            DBInstitution object (created or updated)
        """
        institution_id = institution_data.get('institution_id')
        if not institution_id:
            raise ValueError("institution_id is required for upsert")
        
        existing = await self.get_institution_by_id(institution_id)
        if existing:
            # Update only if new data has more information
            return await self.update_institution(institution_id, institution_data) or existing
        
        return await self.create_institution(institution_data)
