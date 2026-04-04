"""
SJR Data Downloader

Downloads Scimago Journal & Country Rank (SJR) data for specified years
and saves them in the data/ folder with the format: scimagojr_{year}.csv

Scimago provides journal ranking data based on:
- SJR indicator (SCImago Journal Rank)
- H-index
- Citations per document
- Total documents and citations

Usage:
    # Download data for specific years
    python scripts/download_sjr_data.py --years 2022 2023 2024
    
    # Download all available years (1999-2024)
    python scripts/download_sjr_data.py --all
    
    # Download latest year only
    python scripts/download_sjr_data.py --latest
    
    # Download range of years
    python scripts/download_sjr_data.py --range 2020 2024

Source: https://www.scimagojr.com/journalrank.php
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List
import requests
from datetime import datetime

# Add parent directory to path for imports
script_dir = Path(__file__).parent
backend_dir = script_dir.parent
sys.path.insert(0, str(backend_dir))


class SJRDownloader:
    """Downloads SJR data from Scimago website."""
    
    BASE_URL = "https://www.scimagojr.com/journalrank.php"
    DATA_DIR = backend_dir / "data/scimagojr"
    
    # SJR data is available from 1999 to current year
    MIN_YEAR = 1999
    MAX_YEAR = datetime.now().year
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save CSV files (defaults to data/)
        """
        self.output_dir = output_dir or self.DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_year(self, year: int, overwrite: bool = False) -> bool:
        """
        Download SJR data for a specific year.
        
        Args:
            year: Year to download (1999-current)
            overwrite: Whether to overwrite existing file
        
        Returns:
            True if successful, False otherwise
        """
        if year < self.MIN_YEAR or year > self.MAX_YEAR:
            print(f"❌ Invalid year: {year}. Must be between {self.MIN_YEAR} and {self.MAX_YEAR}")
            return False
        
        filename = f"scimagojr_{year}.csv"
        filepath = self.output_dir / filename
        
        # Check if file already exists
        if filepath.exists() and not overwrite:
            print(f"⏭️  Skipping {year}: File already exists at {filepath}")
            return True
        
        print(f"📥 Downloading SJR data for {year}...")
        
        try:
            # Construct download URL
            # Format: https://www.scimagojr.com/journalrank.php?year=2024&out=xls
            params = {
                'year': year,
                'out': 'xls'  # Download as Excel/CSV format
            }
            
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.scimagojr.com/'
            }
            
            # Make request
            response = requests.get(
                self.BASE_URL,
                params=params,
                headers=headers,
                timeout=60,
                stream=True
            )
            
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/csv' not in content_type and 'application/vnd.ms-excel' not in content_type:
                print(f"⚠️  Warning: Unexpected content type: {content_type}")
            
            # Save to file
            total_size = 0
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            # Validate file
            if total_size < 1000:  # Less than 1KB likely means error page
                print(f"❌ Download failed for {year}: File too small ({total_size} bytes)")
                filepath.unlink()  # Delete invalid file
                return False
            
            # Check if file has expected header
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line.startswith('Rank;'):
                    print(f"❌ Download failed for {year}: Invalid CSV format")
                    print(f"   First line: {first_line[:100]}")
                    filepath.unlink()  # Delete invalid file
                    return False
            
            print(f"✅ Successfully downloaded {year}: {filepath} ({total_size:,} bytes)")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error downloading {year}: {e}")
            if filepath.exists():
                filepath.unlink()  # Clean up partial download
            return False
        except Exception as e:
            print(f"❌ Error downloading {year}: {e}")
            if filepath.exists():
                filepath.unlink()  # Clean up partial download
            return False
    
    def download_years(
        self,
        years: List[int],
        overwrite: bool = False,
        delay: float = 2.0
    ) -> dict:
        """
        Download SJR data for multiple years.
        
        Args:
            years: List of years to download
            overwrite: Whether to overwrite existing files
            delay: Delay between requests (seconds) to be polite
        
        Returns:
            Dictionary with success/failure counts
        """
        results = {'success': [], 'failed': [], 'skipped': []}
        
        print(f"\n🚀 Starting download for {len(years)} year(s)...")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Overwrite existing: {overwrite}")
        print("-" * 60)
        
        for i, year in enumerate(sorted(years)):
            success = self.download_year(year, overwrite=overwrite)
            
            if success:
                results['success'].append(year)
            else:
                results['failed'].append(year)
            
            # Be polite - add delay between requests (except for last one)
            if i < len(years) - 1 and success:
                time.sleep(delay)
        
        print("-" * 60)
        print(f"\n📊 Download Summary:")
        print(f"   ✅ Successful: {len(results['success'])} year(s)")
        if results['success']:
            print(f"      {', '.join(map(str, results['success']))}")
        print(f"   ❌ Failed: {len(results['failed'])} year(s)")
        if results['failed']:
            print(f"      {', '.join(map(str, results['failed']))}")
        
        return results
    
    def download_all(self, overwrite: bool = False) -> dict:
        """
        Download all available years (1999-current).
        
        Args:
            overwrite: Whether to overwrite existing files
        
        Returns:
            Dictionary with success/failure counts
        """
        years = list(range(self.MIN_YEAR, self.MAX_YEAR + 1))
        return self.download_years(years, overwrite=overwrite)
    
    def download_latest(self, overwrite: bool = True) -> bool:
        """
        Download only the latest year.
        
        Args:
            overwrite: Whether to overwrite existing file
        
        Returns:
            True if successful, False otherwise
        """
        return self.download_year(self.MAX_YEAR, overwrite=overwrite)
    
    def download_range(
        self,
        start_year: int,
        end_year: int,
        overwrite: bool = False
    ) -> dict:
        """
        Download a range of years (inclusive).
        
        Args:
            start_year: First year to download
            end_year: Last year to download (inclusive)
            overwrite: Whether to overwrite existing files
        
        Returns:
            Dictionary with success/failure counts
        """
        years = list(range(start_year, end_year + 1))
        return self.download_years(years, overwrite=overwrite)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Download SJR (Scimago Journal Rank) data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific years
  python scripts/download_sjr_data.py --years 2022 2023 2024
  
  # Download all available years
  python scripts/download_sjr_data.py --all
  
  # Download latest year
  python scripts/download_sjr_data.py --latest
  
  # Download range of years
  python scripts/download_sjr_data.py --range 2020 2024
  
  # Overwrite existing files
  python scripts/download_sjr_data.py --years 2024 --overwrite
        """
    )
    
    # Download options (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--years',
        type=int,
        nargs='+',
        metavar='YEAR',
        help='Specific years to download (e.g., 2022 2023 2024)'
    )
    group.add_argument(
        '--all',
        action='store_true',
        help='Download all available years (1999-current)'
    )
    group.add_argument(
        '--latest',
        action='store_true',
        help='Download only the latest year'
    )
    group.add_argument(
        '--range',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help='Download range of years (inclusive, e.g., 2020 2024)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (defaults to data/)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between requests in seconds (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = SJRDownloader(output_dir=args.output_dir)
    
    # Execute download based on arguments
    try:
        if args.years:
            downloader.download_years(
                args.years,
                overwrite=args.overwrite,
                delay=args.delay
            )
        elif args.all:
            print("⚠️  Warning: This will download ~25 years of data.")
            confirm = input("Continue? (y/N): ")
            if confirm.lower() == 'y':
                downloader.download_all(overwrite=args.overwrite)
            else:
                print("Cancelled.")
        elif args.latest:
            downloader.download_latest(overwrite=args.overwrite)
        elif args.range:
            start, end = args.range
            downloader.download_range(
                start,
                end,
                overwrite=args.overwrite
            )
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
