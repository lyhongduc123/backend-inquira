"""
Service for institution data enrichment from OpenAlex API.
Handles extraction, transformation, and persistence of institution data.
"""
from typing import Any, Dict, Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.institutions import DBInstitution
from app.extensions.logger import create_logger

from .repository import InstitutionRepository

logger = create_logger(__name__)


class InstitutionService:
    """Service for managing institution data from OpenAlex"""
    
    def __init__(self, db: AsyncSession, repository: Optional["InstitutionRepository"] = None):
        self.db = db
        if repository is None:
            repository = InstitutionRepository(db)
        self.repository = repository
    
    def extract_institution_id_from_url(self, url: str) -> str:
        """Extract OpenAlex institution ID from URL (e.g., https://openalex.org/I170203145 -> I170203145)"""
        if not url:
            return ""
        return url.split("/")[-1] if "/" in url else url
    
    def extract_ror_id(self, ror_url: str) -> Optional[str]:
        """Extract ROR ID from URL (e.g., https://ror.org/00tw3jy02 -> 00tw3jy02)"""
        if not ror_url:
            return None
        return ror_url.split("/")[-1] if "/" in ror_url else ror_url
    
    async def upsert_from_openalex(self, institution: Dict) -> Optional[DBInstitution]:
        """
        Extract and persist institution data from OpenAlex institution object.
        
        Args:
            institution: OpenAlex institution object
            Example:
            {
                "id": "https://openalex.org/I170203145",
                "display_name": "MRC Laboratory of Molecular Biology",
                "ror": "https://ror.org/00tw3jy02",
                "country_code": "GB",
                "country": "United Kingdom",
                "city": "Cambridge",
                "type": "facility",
                "lineage": [...]
            }
            
        Returns:
            DBInstitution object (created or updated) or None if invalid data
        """
        if not institution:
            return None
        
        institution_id_url = institution.get("id")
        if not institution_id_url:
            logger.warning("No institution ID in institution object")
            return None
        
        institution_id = self.extract_institution_id_from_url(institution_id_url)
        display_name = institution.get("display_name", "Unknown Institution")
        
        # Extract ROR ID if available
        ror_url = institution.get("ror")
        ror_id = self.extract_ror_id(ror_url) if ror_url else None
        
        # Extract geographic info - OpenAlex provides country_code, country, city
        country_code = institution.get("country_code")
        country = institution.get("country")  # Full country name
        city = institution.get("city")  # City name
        region = institution.get("region")  # Some institutions have region/state
        
        # Extract institution type
        institution_type = institution.get("type")
        
        # Build external_ids dictionary
        external_ids = {}
        if institution_id_url:
            external_ids["openalex"] = institution_id_url
        if ror_url:
            external_ids["ror"] = ror_url
        
        # Prepare institution data
        institution_data = {
            "institution_id": institution_id,
            "name": display_name,
            "display_name": display_name,
            "ror_id": ror_id,
            "external_ids": external_ids,
            "country_code": country_code,
            "country": country,
            "city": city,
            "region": region,
            "type": institution_type
        }
        
        # Upsert institution
        try:
            db_institution = await self.repository.upsert_institution(institution_data)
            return db_institution
        except Exception as e:
            logger.error(f"Failed to upsert institution {institution_id}: {e}")
            return None
    
    async def batch_upsert_institutions(
        self,
        institutions_data: List[Dict[str, Any]]
    ) -> Dict[str, DBInstitution]:
        """
        Batch upsert multiple institutions in a single database operation.
        More efficient than individual upserts for bulk operations.
        
        Args:
            institutions_data: List of OpenAlex institution objects
            
        Returns:
            Dict mapping institution_id -> DBInstitution object
        """
        if not institutions_data:
            return {}
        
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        
        # Prepare all institution records
        institution_records: List[Dict[str, Any]] = []
        institution_id_map: Dict[str, Dict[str, Any]] = {}
        
        for institution in institutions_data:
            if not institution:
                continue
            
            institution_id_url = institution.get("id")
            if not institution_id_url:
                continue
            
            institution_id = self.extract_institution_id_from_url(institution_id_url)
            display_name = institution.get("display_name", "Unknown Institution")
            
            # Extract ROR ID if available
            ror_url = institution.get("ror")
            ror_id = self.extract_ror_id(ror_url) if ror_url else None
            
            # Extract geographic info
            country_code = institution.get("country_code")
            country = institution.get("country")
            city = institution.get("city")
            region = institution.get("region")
            institution_type = institution.get("type")
            
            # Build external_ids dictionary
            external_ids = {}
            if institution_id_url:
                external_ids["openalex"] = institution_id_url
            if ror_url:
                external_ids["ror"] = ror_url
            
            # Prepare record
            record = {
                "institution_id": institution_id,
                "name": display_name,
                "display_name": display_name,
                "ror_id": ror_id,
                "external_ids": external_ids,
                "country_code": country_code,
                "country": country,
                "city": city,
                "region": region,
                "type": institution_type
            }
            
            institution_records.append(record)
            institution_id_map[institution_id] = institution
        
        if not institution_records:
            return {}
        
        try:
            # Batch insert with ON CONFLICT UPDATE
            stmt = (
                pg_insert(DBInstitution)
                .values(institution_records)
                .on_conflict_do_update(
                    index_elements=['institution_id'],
                    set_={
                        'name': pg_insert(DBInstitution).excluded.name,
                        'display_name': pg_insert(DBInstitution).excluded.display_name,
                        'ror_id': pg_insert(DBInstitution).excluded.ror_id,
                        'external_ids': pg_insert(DBInstitution).excluded.external_ids,
                        'country_code': pg_insert(DBInstitution).excluded.country_code,
                        'country': pg_insert(DBInstitution).excluded.country,
                        'city': pg_insert(DBInstitution).excluded.city,
                        'region': pg_insert(DBInstitution).excluded.region,
                        'type': pg_insert(DBInstitution).excluded.type,
                        'updated_at': datetime.now()
                    }
                )
                .returning(DBInstitution)
            )
            
            result = await self.db.execute(stmt)
            await self.db.commit()
            
            # Build result dict
            db_institutions = result.scalars().all()
            result_map = {inst.institution_id: inst for inst in db_institutions}
            
            logger.info(f"Batch upserted {len(result_map)} institutions")
            return result_map
            
        except Exception as e:
            logger.error(f"Batch upsert institutions failed: {e}", exc_info=True)
            await self.db.rollback()
            return {}
