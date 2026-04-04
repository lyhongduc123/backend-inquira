import asyncio
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.dialects.postgresql import insert

from app.db.database import get_db_session, init_db
from app.models.conferences import DBConference # Nhớ đổi path cho đúng

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COREDataImporter:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def chunked(self, seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    def parse_csv(self) -> List[Dict[str, Any]]:
        records = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Dữ liệu CORE thường dùng dấu phẩy, nhưng đôi khi có ngoặc kép cho tên dài
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 5: continue
                try:
                    # Map theo cấu trúc bạn gửi: 
                    # id, title, acronym, source, rank, is_primary, for1, for2, for3
                    records.append({
                        "core_id": int(row[0]),
                        "title": row[1].strip(),
                        "acronym": row[2].strip() if row[2] else "",
                        "source": row[3].strip(),
                        "rank": row[4].strip() if row[4] else "Unranked",
                        "is_primary": row[5].strip().lower() == 'yes',
                        "for_codes": ",".join(filter(None, row[6:9]))
                    })
                except Exception as e:
                    logger.warning(f"Skip row {row[0] if row else 'unknown'} due to error: {e}")
        return records

    async def run_import(self, update: bool = True):
        records = self.parse_csv()
        logger.info(f"Parsed {len(records)} conferences from {self.file_path.name}")

        async for session in get_db_session():
            try:
                BATCH_SIZE = 100
                for batch in self.chunked(records, BATCH_SIZE):
                    stmt = insert(DBConference).values(batch)
                    
                    if update:
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['title', 'source'],
                            set_={
                                "rank": stmt.excluded.rank,
                                "title": stmt.excluded.title,
                                "core_id": stmt.excluded.core_id,
                                "for_codes": stmt.excluded.for_codes
                            }
                        )
                    else:
                        stmt = stmt.on_conflict_do_nothing()
                    
                    await session.execute(stmt)
                
                await session.commit()
                logger.info("Import CORE data finished successfully!")
            except Exception as e:
                await session.rollback()
                logger.error(f"Database error: {e}")
                raise

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to CORE CSV file')
    args = parser.parse_args()

    await init_db()
    importer = COREDataImporter(Path(args.file))
    await importer.run_import(update=False)

if __name__ == "__main__":
    asyncio.run(main())