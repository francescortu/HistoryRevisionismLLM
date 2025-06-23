"""
EU vs Disinfo Database Scraper

This script scrapes the EU vs Disinfo database to extract structured information
about disinformation cases related to historical revisionism.
"""

import re
import csv
import json
import time
import os
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from typing import List, Dict, Set
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Headers to avoid 403 errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
    "Accept-Encoding": "none",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
    "refere": "https://example.com",
}


def get_page_content(url: str, headers: Dict[str, str] = HEADERS) -> str:
    """
    Function to get the page content while avoiding 403 errors

    Args:
        url: URL to fetch
        headers: HTTP headers to use

    Returns:
        Page content as HTML string
    """
    try:
        req = Request(url, headers=headers)
        response = urlopen(req)
        return response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return ""


def extract_case_links_from_page(html_content: str, base_url: str) -> List[str]:
    """
    Extract individual case links from a database page

    Args:
        html_content: HTML content of the database page
        base_url: Base URL for resolving relative links

    Returns:
        List of case URLs
    """
    soup = BeautifulSoup(html_content, "html.parser")
    case_links = []

    # Find all links that contain "/report/" in the href
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/report/" in href:
            full_url = urljoin(base_url, href)
            case_links.append(full_url)

    return list(set(case_links))  # Remove duplicates


def extract_case_details(html_content: str, case_url: str) -> Dict[str, any]:
    """
    Extract detailed information from a single case page

    Args:
        html_content: HTML content of the case page
        case_url: URL of the case page

    Returns:
        Dictionary containing extracted case information
    """
    soup = BeautifulSoup(html_content, "html.parser")

    case_data = {
        "url": case_url,
        "title": "",
        "summary": "",
        "response": "",
        "outlet": "",
        "date_of_publication": "",
        "countries": [],
        "tags": [],
    }

    try:
        # Extract title and remove DISINFO: prefix
        title_element = soup.find("h1")
        if title_element:
            title_text = title_element.get_text(strip=True)
            # Remove DISINFO: prefix if present
            if title_text.startswith("DISINFO:"):
                title_text = title_text[8:].strip()
            case_data["title"] = title_text

        # Get the full page text for easier processing
        full_text = soup.get_text()

        # Extract summary - look for text between SUMMARY and RESPONSE
        summary_match = re.search(
            r"SUMMARY\s*(.*?)\s*RESPONSE", full_text, re.DOTALL | re.IGNORECASE
        )
        if summary_match:
            case_data["summary"] = summary_match.group(1).strip()

        # Extract response - look for text after RESPONSE
        response_match = re.search(
            r"RESPONSE\s*(.*?)(?=\n\s*(?:Outlet|Date|Countries|TAGS|$))",
            full_text,
            re.DOTALL | re.IGNORECASE,
        )
        if response_match:
            case_data["response"] = response_match.group(1).strip()

        # Extract outlet, date, and countries from the details section
        details_text = soup.get_text()

        # Extract outlet
        outlet_match = re.search(r"Outlet:\s*([^\n*]+)", details_text)
        if outlet_match:
            case_data["outlet"] = outlet_match.group(1).strip()

        # Extract date
        date_match = re.search(r"Date of publication:\s*([^\n]+)", details_text)
        if date_match:
            case_data["date_of_publication"] = date_match.group(1).strip()

        # Extract countries
        countries_match = re.search(
            r"Countries / regions discussed:\s*([^\n]+)", details_text
        )
        if countries_match:
            countries_text = countries_match.group(1).strip()
            case_data["countries"] = [c.strip() for c in countries_text.split(",")]

        # Extract tags
        tags_section = soup.find("div", class_="tags") or soup.find_all(
            "a", href=lambda href: href and "/report/tag/" in href
        )
        if tags_section:
            if isinstance(tags_section, list):
                for tag_link in tags_section:
                    case_data["tags"].append(tag_link.get_text(strip=True))
            else:
                for tag_link in tags_section.find_all("a"):
                    case_data["tags"].append(tag_link.get_text(strip=True))

        # Alternative tag extraction from TAGS: section
        if not case_data["tags"]:
            tags_match = re.search(r"TAGS:\s*(.+?)(?=\n|$)", details_text, re.DOTALL)
            if tags_match:
                tags_text = tags_match.group(1)
                # Extract tag names from brackets
                tag_matches = re.findall(r"\[([^\]]+)\]", tags_text)
                case_data["tags"] = tag_matches

    except Exception as e:
        logger.error(f"Error extracting details from {case_url}: {e}")

    return case_data


def get_total_pages(html_content: str) -> int:
    """
    Extract the total number of pages from pagination

    Args:
        html_content: HTML content of the first page

    Returns:
        Total number of pages
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Look for pagination links
    page_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/page/" in href and "disinfo_keywords" in href:
            page_match = re.search(r"/page/(\d+)/", href)
            if page_match:
                page_links.append(int(page_match.group(1)))

    return max(page_links) if page_links else 1


def load_processed_urls(checkpoint_file: str) -> Set[str]:
    """
    Load already processed URLs from checkpoint file

    Args:
        checkpoint_file: Path to checkpoint file

    Returns:
        Set of already processed URLs
    """
    processed_urls = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    url = line.strip()
                    if url:
                        processed_urls.add(url)
            logger.info(
                f"Loaded {len(processed_urls)} already processed URLs from checkpoint"
            )
        except Exception as e:
            logger.error(f"Error loading checkpoint file: {e}")
    return processed_urls


def save_checkpoint(processed_urls: Set[str], checkpoint_file: str) -> None:
    """
    Save processed URLs to checkpoint file

    Args:
        processed_urls: Set of processed URLs
        checkpoint_file: Path to checkpoint file
    """
    try:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            for url in sorted(processed_urls):
                f.write(url + "\n")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def append_to_csv(cases: List[Dict[str, any]], filename: str) -> None:
    """
    Append cases to CSV file

    Args:
        cases: List of case dictionaries
        filename: Output filename
    """
    if not cases:
        return

    fieldnames = [
        "url",
        "title",
        "summary",
        "response",
        "outlet",
        "date_of_publication",
        "countries",
        "tags",
    ]

    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        for case in cases:
            # Convert lists to strings for CSV
            row = case.copy()
            row["countries"] = "; ".join(case["countries"])
            row["tags"] = "; ".join(case["tags"])
            writer.writerow(row)


def scrape_euvsdisinfo_database(
    base_url: str, output_file: str = "euvsdisinfo_cases.csv", delay: float = 1.0
) -> List[Dict[str, any]]:
    """
    Main function to scrape the entire EU vs Disinfo database with checkpointing

    Args:
        base_url: Base URL of the database
        output_file: Output CSV file name
        delay: Delay between requests in seconds

    Returns:
        List of all extracted cases
    """
    logger.info("Starting EU vs Disinfo database scraping...")

    # Setup checkpoint files
    checkpoint_file = output_file.replace(".csv", "_checkpoint.txt")
    processed_urls = load_processed_urls(checkpoint_file)

    # Get first page to determine total pages
    logger.info("Fetching first page to determine pagination...")
    first_page_content = get_page_content(base_url)
    if not first_page_content:
        logger.error("Failed to fetch first page")
        return []

    total_pages = get_total_pages(first_page_content)
    logger.info(f"Found {total_pages} pages to scrape")

    all_case_links = []

    # Collect all case links from all pages
    for page_num in range(1, total_pages + 1):
        if page_num == 1:
            page_url = base_url
            page_content = first_page_content
        else:
            page_url = f"{base_url.split('?')[0]}/page/{page_num}/?disinfo_keywords[]=keyword_77395&sort=asc"
            logger.info(f"Fetching page {page_num}/{total_pages}: {page_url}")
            page_content = get_page_content(page_url)
            time.sleep(delay)  # Be respectful to the server

        if page_content:
            case_links = extract_case_links_from_page(page_content, base_url)
            all_case_links.extend(case_links)
            logger.info(f"Found {len(case_links)} cases on page {page_num}")
        else:
            logger.warning(f"Failed to fetch page {page_num}")

    # Remove duplicates and sort
    all_case_links = list(set(all_case_links))
    all_case_links.sort()
    logger.info(f"Total unique cases found: {len(all_case_links)}")

    # Filter out already processed cases
    remaining_case_links = [url for url in all_case_links if url not in processed_urls]
    logger.info(f"Already processed: {len(processed_urls)} cases")
    logger.info(f"Remaining to process: {len(remaining_case_links)} cases")

    # Extract details from each case with checkpointing every 10 cases
    batch_cases = []
    total_processed = len(processed_urls)

    for i, case_url in enumerate(remaining_case_links, 1):
        current_case_num = total_processed + i
        logger.info(
            f"Processing case {current_case_num}/{len(all_case_links)}: {case_url}"
        )

        case_content = get_page_content(case_url)
        if case_content:
            case_data = extract_case_details(case_content, case_url)
            batch_cases.append(case_data)
            processed_urls.add(case_url)

            # Save checkpoint every 10 cases
            if i % 10 == 0:
                # Append to CSV file
                append_to_csv(batch_cases, output_file)
                logger.info(f"Saved batch of {len(batch_cases)} cases to {output_file}")

                # Update checkpoint file
                save_checkpoint(processed_urls, checkpoint_file)
                logger.info(
                    f"Updated checkpoint after processing {current_case_num} cases"
                )

                # Clear batch
                batch_cases = []
        else:
            logger.warning(f"Failed to fetch case: {case_url}")

        time.sleep(delay)  # Be respectful to the server

    # Save any remaining cases in the final batch
    if batch_cases:
        append_to_csv(batch_cases, output_file)
        save_checkpoint(processed_urls, checkpoint_file)
        logger.info(f"Saved final batch of {len(batch_cases)} cases")

    # Load all processed cases for final JSON export
    all_processed_cases = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert back to lists
                    row["countries"] = [
                        c.strip() for c in row["countries"].split(";") if c.strip()
                    ]
                    row["tags"] = [
                        t.strip() for t in row["tags"].split(";") if t.strip()
                    ]
                    all_processed_cases.append(row)
        except Exception as e:
            logger.error(f"Error reading final CSV: {e}")

    # Save final JSON
    if all_processed_cases:
        save_to_json(all_processed_cases, output_file.replace(".csv", ".json"))

    logger.info(f"Scraping completed! Total cases processed: {len(processed_urls)}")
    return all_processed_cases


def save_to_csv(cases: List[Dict[str, any]], filename: str) -> None:
    """
    Save cases to CSV file

    Args:
        cases: List of case dictionaries
        filename: Output filename
    """
    if not cases:
        return

    fieldnames = [
        "url",
        "title",
        "summary",
        "response",
        "outlet",
        "date_of_publication",
        "countries",
        "tags",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            # Convert lists to strings for CSV
            row = case.copy()
            row["countries"] = "; ".join(case["countries"])
            row["tags"] = "; ".join(case["tags"])
            writer.writerow(row)

    logger.info(f"Saved {len(cases)} cases to {filename}")


def save_to_json(cases: List[Dict[str, any]], filename: str) -> None:
    """
    Save cases to JSON file

    Args:
        cases: List of case dictionaries
        filename: Output filename
    """
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(cases, jsonfile, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(cases)} cases to {filename}")


def main():
    """Main execution function"""
    base_url = "https://euvsdisinfo.eu/disinformation-cases/?disinfo_keywords[]=keyword_77395&sort=asc"

    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    output_csv = "data/euvsdisinfo_historical_revisionism_cases.csv"

    # Start scraping with checkpointing
    cases = scrape_euvsdisinfo_database(base_url, output_csv, delay=1.5)

    # Print summary
    print("\n=== SCRAPING SUMMARY ===")
    print(f"Total cases extracted: {len(cases)}")
    print("Output files:")
    print(f"  - CSV: {output_csv}")
    print(f"  - JSON: {output_csv.replace('.csv', '.json')}")
    print(f"  - Checkpoint: {output_csv.replace('.csv', '_checkpoint.txt')}")

    if cases:
        print("\nSample case:")
        sample = cases[0]
        for key, value in sample.items():
            if isinstance(value, list):
                print(
                    f"  {key}: {', '.join(value[:3])}{'...' if len(value) > 3 else ''}"
                )
            else:
                print(f"  {key}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")


if __name__ == "__main__":
    main()
