"""
================================================================================
Shadow Supply Chain Hunter - SEC 10-K Ingestion Pipeline
================================================================================
Module: ingest.py
Purpose: Download and preprocess SEC 10-K filings for target companies

QUANT OPTIMIZATION NOTES:
- We pull only the LATEST filing (limit=1) to ensure our graph reflects current
  supply chain relationships. Historical analysis would require time-series graph
  versioning which adds complexity without improving signal quality for risk assessment.
- 10-K filings contain "Risk Factors" (Item 1A) and "Business" (Item 1) sections
  which are gold mines for supply chain dependency extraction.
================================================================================
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEC_EMAIL = os.getenv("SEC_EDGAR_EMAIL", "your_email@example.com")

# Target companies for supply chain analysis
# Strategic selection: AAPL (consumer tech), NVDA (AI chips), TSM (foundry)
# This creates a natural supply chain: TSM -> NVDA -> AAPL
TARGET_TICKERS = ["AAPL", "NVDA", "TSM"]


def setup_data_directory() -> Path:
    """
    Initialize the data directory structure.
    
    Returns:
        Path to the data directory
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Data directory initialized at: {DATA_DIR}")
    return DATA_DIR


def download_filings(
    ticker: str,
    filing_type: str = "10-K",
    limit: int = 1,
    include_amends: bool = False
) -> Optional[Path]:
    """
    Download SEC filings for a given ticker.
    
    QUANT RATIONALE:
    - limit=1: We want the LATEST snapshot of supply chain dependencies.
      Older filings create noise as supplier relationships evolve.
    - include_amends=False: Amendments typically fix accounting issues,
      not supply chain disclosures. Skip for cleaner data.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        filing_type: SEC filing type (default: '10-K' for annual reports)
        limit: Number of filings to download (default: 1 for latest only)
        include_amends: Whether to include amended filings (default: False)
    
    Returns:
        Path to downloaded filing directory, or None if failed
    """
    print(f"\n[DOWNLOAD] Fetching {filing_type} for {ticker}...")
    
    # Initialize SEC downloader with required email
    # SEC requires identification for rate limiting compliance
    dl = Downloader(company_name="ShadowSupplyChainHunter", email_address=SEC_EMAIL)
    
    try:
        # Download to our data directory
        dl.get(
            filing_type,
            ticker,
            limit=limit,
            download_details=True,  # Get full filing package
            include_amends=include_amends
        )
        
        # Locate the downloaded files
        # sec-edgar-downloader saves to: ./sec-edgar-filings/{ticker}/{filing_type}/
        default_download_path = Path("./sec-edgar-filings") / ticker / filing_type
        
        if default_download_path.exists():
            # Move to our data directory for cleaner organization
            target_path = DATA_DIR / ticker
            if target_path.exists():
                shutil.rmtree(target_path)  # Clean previous downloads
            shutil.move(str(default_download_path), str(target_path))
            
            # Cleanup the sec-edgar-filings directory
            sec_filings_dir = Path("./sec-edgar-filings")
            if sec_filings_dir.exists():
                shutil.rmtree(sec_filings_dir)
            
            print(f"[SUCCESS] {ticker} 10-K saved to: {target_path}")
            return target_path
        else:
            print(f"[WARNING] No filings found for {ticker}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to download {ticker}: {str(e)}")
        return None


def download_all_targets(tickers: List[str] = TARGET_TICKERS) -> dict:
    """
    Download 10-K filings for all target companies.
    
    QUANT OPTIMIZATION:
    - Sequential downloads to respect SEC rate limits (10 req/sec)
    - Progress tracking for pipeline observability
    
    Args:
        tickers: List of ticker symbols to download
    
    Returns:
        Dictionary mapping tickers to their download paths
    """
    setup_data_directory()
    
    results = {}
    print(f"\n{'='*60}")
    print(f"SHADOW SUPPLY CHAIN HUNTER - DATA INGESTION")
    print(f"{'='*60}")
    print(f"Targets: {', '.join(tickers)}")
    print(f"Filing Type: 10-K (Annual Report)")
    print(f"{'='*60}\n")
    
    for ticker in tqdm(tickers, desc="Downloading 10-K filings"):
        path = download_filings(ticker)
        results[ticker] = path
    
    # Summary report
    print(f"\n{'='*60}")
    print("INGESTION SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for p in results.values() if p is not None)
    print(f"Successfully downloaded: {successful}/{len(tickers)}")
    
    for ticker, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {ticker}: {path or 'FAILED'}")
    
    return results


def get_filing_text_files() -> List[Path]:
    """
    Retrieve all text-based filing documents from the data directory.
    
    QUANT RATIONALE:
    - We prioritize .txt and .htm files as they contain the actual filing text
    - PDFs require OCR which introduces extraction errors
    - The 'full-submission.txt' contains the complete filing in one document
    
    Returns:
        List of paths to filing text files
    """
    text_files = []
    
    for ticker_dir in DATA_DIR.iterdir():
        if ticker_dir.is_dir():
            # Look for filing documents
            for filing_dir in ticker_dir.rglob("*"):
                if filing_dir.is_file():
                    # Prefer full-submission.txt for complete coverage
                    if filing_dir.suffix.lower() in ['.txt', '.htm', '.html']:
                        text_files.append(filing_dir)
    
    # Deduplicate and prefer full-submission files
    full_submissions = [f for f in text_files if 'full-submission' in f.name.lower()]
    
    if full_submissions:
        print(f"[INFO] Found {len(full_submissions)} full-submission files (preferred)")
        return full_submissions
    
    print(f"[INFO] Found {len(text_files)} filing documents")
    return text_files


if __name__ == "__main__":
    """
    Main execution: Download all target company 10-K filings
    
    Usage:
        python src/ingest.py
    
    This will download the latest 10-K for AAPL, NVDA, and TSM
    to the /data directory for subsequent graph construction.
    """
    results = download_all_targets()
    
    # List available files for the graph pipeline
    print(f"\n{'='*60}")
    print("FILES READY FOR GRAPH CONSTRUCTION")
    print(f"{'='*60}")
    files = get_filing_text_files()
    for f in files:
        print(f"  → {f.relative_to(PROJECT_ROOT)}")
