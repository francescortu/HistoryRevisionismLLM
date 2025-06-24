#!/usr/bin/env python3
"""
JSON Cleaner for EUvsDisinfo Historical Revisionism Cases

This script cleans and postprocesses the scraped JSON data by:
1. Removing empty entries
2. Cleaning HTML artifacts and navigation elements
3. Standardizing field formats
4. Removing invalid or incomplete entries
"""

import json
import re
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class EUvsDisinfoJSONCleaner:
    def __init__(self):
        # Patterns to identify and remove web artifacts
        self.web_artifacts = [
            r"Also recommended",
            r"Related disinfo cases",
            r"Database\n\nDISINFO:",
            r"MORE\n\n\n\n\n\n\n\n\nDON\'T BE DECEIVED",
            r"Contact us European External Action Service",
            r"ContentArticles\nDatabase\nLearn",
            r"CategoriesUkraine\nBelarus\nMoldova",
            r"COOKIES\nDATA PROTECTION\nDISCLAIMER",
            r"Your opinion matters!",
            r"Subscribe to the Disinfo Review",
            r"Data Protection Information",
            r"I have read and understood the Privacy Statement",
            r"SEND COMMENT",
            r"SUBSCRIBE",
            r"English\t\t\t\t\t\t\t\t",
            r"Accessibility\s+Adjustments",
            r"Powered by\n\n\t\t\t\t\t\t\t\t\t\t\tOneTap",
            r"Version \d+\.\d+\.\d+",
            r"\n\n\n+",  # Multiple newlines
            r"\t+",  # Multiple tabs
        ]

        # Patterns for navigation and UI elements
        self.ui_elements = [
            r"article\n\n\n\n\n\n",
            r"case\n\n\nDISINFO:",
            r"video\n\n\n\n\n\n",
            r"infographic\n\n\n\n\n",
            r"Bigger Text\n\n\n",
            r"Cursor\n\n\n",
            r"Letter Spacing\n\n\n",
            r"Readable Font\n\n\n",
            r"Line Height\n\n\n",
            r"Colors\t\t\t\t\t\t\t\t\n",
            r"Grayscale\n\n\n",
            r"Brightness\n\n\n",
            r"Hide Images\n\n\n",
            r"Reading Mask\n\n\n",
            r"Reset Settings\t\t\t\t\t\t\n",
        ]

        # Required fields for a valid entry
        self.required_fields = ["url", "title", "summary", "response"]

    def clean_text(self, text: str) -> str:
        """Clean text from web artifacts and formatting issues."""
        if not isinstance(text, str):
            return text

        # Truncation patterns - everything after these should be removed
        truncation_patterns = [
            r"Also recommended",
            r"Related articles",
            r"MORE\s*DON\'T BE DECEIVED",
            r"MOREDON\'T BE DECEIVED",  # No space version
            r"Contact us European External Action Service",
            r"ContentArticles\s*Database\s*Learn",
            r"CategoriesUkraine\s*Belarus\s*Moldova",
            r"Your opinion matters!",
            r"Subscribe to the Disinfo Review",
            r"Data Protection Information",
            r"COOKIES\s*DATA PROTECTION\s*DISCLAIMER",
            r"Disclaimer\s*Cases in the EUvsDisinfo database",
            r"English\s*Deutsch\s*Español",  # Language selector
            r"Accessibility\s*Adjustments",
            r"Powered by\s*OneTap",
            r"Version \d+\.\d+\.\d+",
            # Navigation section patterns - these indicate start of navigation/related content
            r"\n\n\n+article\n",  # Article in navigation format
            r"\n\n\n+video\n",  # Video in navigation format
            r"\n\n\n+infographic\n",  # Infographic in navigation format
            r"\n\n\n+case\n",  # Case in navigation format
            r"caseDISINFO:",
            # EEAS contact info
            r"\(EEAS\)",
            r"General enquiries",
            r"EEAS press team",
            # Language menu (usually at bottom)
            r"DeutschEspañolFrançais",
        ]

        # Find the earliest truncation point
        truncation_pos = len(text)
        for pattern in truncation_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                truncation_pos = min(truncation_pos, match.start())

        # Truncate text if truncation point found
        if truncation_pos < len(text):
            text = text[:truncation_pos].strip()

        # Remove remaining web artifacts
        for pattern in self.web_artifacts + self.ui_elements:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
        text = re.sub(r"\t+", " ", text)  # Replace tabs with single space
        text = re.sub(r" {2,}", " ", text)  # Replace multiple spaces with single space
        text = text.strip()

        return text

    def is_valid_entry(self, entry: Dict[str, Any]) -> bool:
        """Check if an entry is valid and contains meaningful data."""
        # Empty entry
        if not entry:
            return False

        # Check for required fields
        has_required = any(field in entry for field in self.required_fields)
        if not has_required:
            return False

        # Check if entry has meaningful content
        if "summary" in entry:
            summary = entry["summary"]
            if isinstance(summary, str):
                # Remove artifacts and check if meaningful content remains
                cleaned_summary = self.clean_text(summary)
                if len(cleaned_summary) < 10:  # Too short to be meaningful
                    return False

                # Check for artifact-only content
                artifact_indicators = [
                    "also recommended",
                    "related disinfo cases",
                    "database",
                    "don't be deceived",
                    "contact us european",
                    "version 2.2.0",
                ]

                if any(
                    indicator in cleaned_summary.lower()
                    for indicator in artifact_indicators
                ):
                    return False

        return True

    def clean_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean a single entry."""
        if not self.is_valid_entry(entry):
            return None

        cleaned_entry = {}

        for key, value in entry.items():
            if isinstance(value, str):
                cleaned_value = self.clean_text(value)
                # Only include if there's meaningful content
                if cleaned_value and len(cleaned_value.strip()) > 0:
                    cleaned_entry[key] = cleaned_value
            elif isinstance(value, list):
                # Clean list items if they're strings
                cleaned_list = []
                for item in value:
                    if isinstance(item, str):
                        cleaned_item = self.clean_text(item)
                        if cleaned_item and len(cleaned_item.strip()) > 0:
                            cleaned_list.append(cleaned_item)
                    else:
                        cleaned_list.append(item)
                if cleaned_list:  # Only include non-empty lists
                    cleaned_entry[key] = cleaned_list
            else:
                cleaned_entry[key] = value

        return cleaned_entry if cleaned_entry else None

    def clean_json_file(self, input_file: Path, output_file: Path) -> None:
        """Clean the entire JSON file."""
        logger.info(f"Loading JSON file: {input_file}")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file: {e}")
            return
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return

        if not isinstance(data, list):
            logger.error("Expected JSON file to contain a list")
            return

        logger.info(f"Original entries: {len(data)}")

        # Clean entries
        cleaned_entries = []
        empty_count = 0
        artifact_count = 0

        for i, entry in enumerate(data):
            if not entry:  # Empty entry
                empty_count += 1
                continue

            cleaned_entry = self.clean_entry(entry)
            if cleaned_entry is None:
                artifact_count += 1
                logger.debug(
                    f"Removed entry {i}: contains only artifacts or invalid data"
                )
            else:
                cleaned_entries.append(cleaned_entry)

        logger.info(f"Cleaned entries: {len(cleaned_entries)}")
        logger.info(f"Removed {empty_count} empty entries")
        logger.info(f"Removed {artifact_count} entries with artifacts/invalid data")

        # Save cleaned data
        logger.info(f"Saving cleaned data to: {output_file}")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_entries, f, indent=2, ensure_ascii=False)
            logger.success(f"Successfully saved {len(cleaned_entries)} cleaned entries")
        except Exception as e:
            logger.error(f"Failed to save cleaned file: {e}")

    def validate_cleaned_data(self, file_path: Path) -> None:
        """Validate the cleaned data and report statistics."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read cleaned file: {e}")
            return

        logger.info("Validation of cleaned data:")
        logger.info(f"Total entries: {len(data)}")

        # Count entries by field presence
        field_counts = {}
        for entry in data:
            for field in [
                "url",
                "title",
                "summary",
                "response",
                "outlet",
                "date_of_publication",
                "countries",
                "tags",
            ]:
                if field in entry and entry[field]:
                    field_counts[field] = field_counts.get(field, 0) + 1

        logger.info("Field presence statistics:")
        for field, count in sorted(field_counts.items()):
            percentage = (count / len(data)) * 100
            logger.info(f"  {field}: {count}/{len(data)} ({percentage:.1f}%)")


def main():
    """Main function to clean the JSON file."""
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    input_file = data_dir / "euvsdisinfo_historical_revisionism_cases.json"
    output_file = data_dir / "euvsdisinfo_historical_revisionism_cases_cleaned.json"
    backup_file = data_dir / "euvsdisinfo_historical_revisionism_cases_backup.json"

    # Initialize cleaner
    cleaner = EUvsDisinfoJSONCleaner()

    # Create backup
    if input_file.exists():
        logger.info(f"Creating backup: {backup_file}")
        import shutil

        shutil.copy2(input_file, backup_file)

    # Clean the file
    cleaner.clean_json_file(input_file, output_file)

    # Validate cleaned data
    if output_file.exists():
        cleaner.validate_cleaned_data(output_file)

        # Ask user if they want to replace the original
        logger.info(f"Cleaned file saved as: {output_file}")
        logger.info(f"Original file backed up as: {backup_file}")
        logger.info("Review the cleaned file and replace the original if satisfied.")


if __name__ == "__main__":
    main()
