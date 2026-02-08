"""
SJR Data Import Script

Imports SCImago Journal & Country Rank data from CSV files into the database.
Supports initial import and incremental updates.

Usage:
    # Import all years
    python scripts/import_sjr_data.py --all
    
    # Import specific year
    python scripts/import_sjr_data.py --year 2024
    
    # Update existing data (upsert)
    python scripts/import_sjr_data.py --all --update
    
    # Dry run (no database changes)
    python scripts/import_sjr_data.py --all --dry-run
"""

import asyncio
import csv
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.database import get_db_session, init_db
from app.models.journals import DBJournal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SJRDataImporter:
    """Handles import of SJR journal data from CSV files."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the importer.
        
        Args:
            data_dir: Directory containing SJR CSV files. Defaults to backend-exegent/data/
        """
        if data_dir is None:
            # Default to data directory in project root
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def normalize_title(self, title: str) -> str:
        """
        Normalize journal title for fuzzy matching.
        
        - Convert to lowercase
        - Remove punctuation
        - Remove extra whitespace
        """
        if not title:
            return ""
        
        # Lowercase
        normalized = title.lower()
        
        # Remove punctuation except spaces and hyphens
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def parse_float(self, value: str) -> Optional[float]:
        """Parse float value, handling European decimal format (comma as decimal separator)."""
        if not value or value.strip() == '':
            return None
        try:
            # Replace comma with dot for European format
            value = value.replace(',', '.')
            return float(value)
        except ValueError:
            return None
    
    def parse_int(self, value: str) -> Optional[int]:
        """Parse integer value."""
        if not value or value.strip() == '':
            return None
        try:
            return int(value)
        except ValueError:
            return None
    
    def parse_boolean(self, value: str) -> bool:
        """Parse boolean value from Yes/No."""
        return value.strip().lower() == 'yes'
    
    def parse_array(self, value: str) -> List[str]:
        """Parse semicolon or comma-separated values into array."""
        if not value or value.strip() == '':
            return []
        
        # Split by semicolon first, then comma
        items = []
        if ';' in value:
            items = [item.strip() for item in value.split(';') if item.strip()]
        elif ',' in value:
            items = [item.strip() for item in value.split(',') if item.strip()]
        else:
            items = [value.strip()]
        
        return items
    
    def parse_csv_row(self, row: Dict[str, str], year: int) -> Dict[str, Any]:
        """
        Parse a single CSV row into database fields.
        
        Args:
            row: Dictionary from CSV DictReader
            year: Year of the data
        
        Returns:
            Dictionary with parsed fields for DBJournal
        """
        title = row.get('Title', '').strip()
        
        # Build search terms for fuzzy matching
        search_terms_parts = [
            title,
            row.get('Publisher', ''),
            row.get('Categories', ''),
            row.get('Areas', '')
        ]
        search_terms = ' '.join(filter(None, search_terms_parts))
        
        return {
            'source_id': row.get('Sourceid', '').strip(),
            'title': title,
            'title_normalized': self.normalize_title(title),
            'type': row.get('Type', 'journal').strip().lower(),
            'issn': row.get('Issn', '').strip() or None,
            'publisher': row.get('Publisher', '').strip() or None,
            'country': row.get('Country', '').strip() or None,
            'region': row.get('Region', '').strip() or None,
            'coverage': row.get('Coverage', '').strip() or None,
            'is_open_access': self.parse_boolean(row.get('Open Access', 'No')),
            'is_open_access_diamond': self.parse_boolean(row.get('Open Access Diamond', 'No')),
            'sjr_score': self.parse_float(row.get('SJR', '')),
            'sjr_best_quartile': row.get('SJR Best Quartile', '').strip() or None,
            'h_index': self.parse_int(row.get('H index', '')),
            'total_docs_current_year': self.parse_int(row.get(f'Total Docs. ({year})', '')),
            'total_docs_3years': self.parse_int(row.get('Total Docs. (3years)', '')),
            'total_refs': self.parse_int(row.get('Total Refs.', '')),
            'total_cites_3years': self.parse_int(row.get('Total Citations (3years)', '')),
            'citable_docs_3years': self.parse_int(row.get('Citable Docs. (3years)', '')),
            'cites_per_doc_2years': self.parse_float(row.get('Citations / Doc. (2years)', '')),
            'refs_per_doc': self.parse_float(row.get('Ref. / Doc.', '')),
            'percent_female': self.parse_float(row.get('%Female', '')),
            'overton_count': self.parse_int(row.get('Overton', '')),
            'sdg_count': self.parse_int(row.get('SDG', '')),
            'categories': self.parse_array(row.get('Categories', '')),
            'areas': self.parse_array(row.get('Areas', '')),
            'data_year': year,
            'rank': self.parse_int(row.get('Rank', '')),
            'search_terms': search_terms.lower(),
        }
    
    async def import_csv_file(
        self, 
        file_path: Path, 
        year: int,
        dry_run: bool = False,
        update: bool = False
    ) -> int:
        """
        Import data from a single CSV file.
        
        Args:
            file_path: Path to CSV file
            year: Year of the data
            dry_run: If True, don't commit to database
            update: If True, update existing records (upsert)
        
        Returns:
            Number of records processed
        """
        logger.info(f"Processing {file_path} for year {year}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 0
        
        records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # SJR uses semicolon as delimiter
            reader = csv.DictReader(f, delimiter=';')
            
            for row in reader:
                try:
                    record = self.parse_csv_row(row, year)
                    records.append(record)
                except Exception as e:
                    logger.error(f"Error parsing row: {e}")
                    logger.debug(f"Problematic row: {row}")
                    continue
        
        logger.info(f"Parsed {len(records)} records from {file_path.name}")
        
        if dry_run:
            logger.info("DRY RUN - No database changes made")
            return len(records)
        
        # Import to database
        async for session in get_db_session():
            try:
                if update:
                    # Upsert: Insert or update on conflict
                    stmt = insert(DBJournal).values(records)
                    stmt = stmt.on_conflict_do_update(
                        constraint='uq_journal_source_year',
                        set_={
                            'title': stmt.excluded.title,
                            'title_normalized': stmt.excluded.title_normalized,
                            'type': stmt.excluded.type,
                            'issn': stmt.excluded.issn,
                            'publisher': stmt.excluded.publisher,
                            'country': stmt.excluded.country,
                            'region': stmt.excluded.region,
                            'coverage': stmt.excluded.coverage,
                            'is_open_access': stmt.excluded.is_open_access,
                            'is_open_access_diamond': stmt.excluded.is_open_access_diamond,
                            'sjr_score': stmt.excluded.sjr_score,
                            'sjr_best_quartile': stmt.excluded.sjr_best_quartile,
                            'h_index': stmt.excluded.h_index,
                            'total_docs_current_year': stmt.excluded.total_docs_current_year,
                            'total_docs_3years': stmt.excluded.total_docs_3years,
                            'total_refs': stmt.excluded.total_refs,
                            'total_cites_3years': stmt.excluded.total_cites_3years,
                            'citable_docs_3years': stmt.excluded.citable_docs_3years,
                            'cites_per_doc_2years': stmt.excluded.cites_per_doc_2years,
                            'refs_per_doc': stmt.excluded.refs_per_doc,
                            'percent_female': stmt.excluded.percent_female,
                            'overton_count': stmt.excluded.overton_count,
                            'sdg_count': stmt.excluded.sdg_count,
                            'categories': stmt.excluded.categories,
                            'areas': stmt.excluded.areas,
                            'rank': stmt.excluded.rank,
                            'search_terms': stmt.excluded.search_terms,
                            'updated_at': stmt.excluded.updated_at,
                        }
                    )
                    await session.execute(stmt)
                    logger.info(f"Upserted {len(records)} records for year {year}")
                else:
                    # Insert only (will fail on conflicts)
                    session.add_all([DBJournal(**record) for record in records])
                    logger.info(f"Inserted {len(records)} records for year {year}")
                
                await session.commit()
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Database error: {e}")
                raise
        
        return len(records)
    
    async def import_all_years(
        self,
        years: Optional[List[int]] = None,
        dry_run: bool = False,
        update: bool = False
    ) -> Dict[int, int]:
        """
        Import data for all available years.
        
        Args:
            years: List of years to import. If None, imports all found CSV files.
            dry_run: If True, don't commit to database
            update: If True, update existing records (upsert)
        
        Returns:
            Dictionary mapping year to number of records imported
        """
        if years is None:
            # Find all SJR CSV files
            csv_files = list(self.data_dir.glob('scimagojr_*.csv'))
            years = []
            for csv_file in csv_files:
                match = re.search(r'scimagojr_(\d{4})\.csv', csv_file.name)
                if match:
                    years.append(int(match.group(1)))
            years.sort()
        
        logger.info(f"Importing data for years: {years}")
        
        results = {}
        for year in years:
            csv_file = self.data_dir / f'scimagojr_{year}.csv'
            count = await self.import_csv_file(csv_file, year, dry_run, update)
            results[year] = count
        
        return results


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Import SJR journal data into database')
    parser.add_argument('--all', action='store_true', help='Import all available years')
    parser.add_argument('--year', type=int, help='Import specific year')
    parser.add_argument('--update', action='store_true', help='Update existing records (upsert)')
    parser.add_argument('--dry-run', action='store_true', help='Parse files but don\'t write to database')
    parser.add_argument('--data-dir', type=str, help='Custom data directory path')
    
    args = parser.parse_args()
    
    if not args.all and not args.year:
        parser.error('Must specify either --all or --year')
    
    # Initialize database
    await init_db()
    
    # Create importer
    importer = SJRDataImporter(data_dir=args.data_dir)
    
    # Import data
    if args.all:
        results = await importer.import_all_years(dry_run=args.dry_run, update=args.update)
        logger.info("=" * 60)
        logger.info("Import Summary:")
        for year, count in results.items():
            logger.info(f"  {year}: {count} records")
        logger.info(f"Total: {sum(results.values())} records")
    else:
        count = await importer.import_csv_file(
            importer.data_dir / f'scimagojr_{args.year}.csv',
            args.year,
            dry_run=args.dry_run,
            update=args.update
        )
        logger.info(f"Imported {count} records for {args.year}")
    
    logger.info("Done!")


if __name__ == '__main__':
    asyncio.run(main())
