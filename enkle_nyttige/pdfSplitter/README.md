# PDF Section Splitter

A flexible Python script for splitting PDFs into named sections based on page numbers. Perfect for extracting chapters, parts, or any arbitrary sections from large PDF documents.

## Features

- **Simple page-based splitting**: Specify start pages for each section
- **Preserves metadata**: Keeps document metadata, bookmarks, and viewer preferences
- **Clean naming**: Automatically generates numbered, sanitized filenames
- **Error handling**: Validates inputs and provides clear error messages
- **Non-destructive**: Original file is never modified

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install pypdf pytest
   ```

2. **Make the script executable** (optional):
   ```bash
   chmod +x split_pdf_sections.py
   ```

## Usage

### Basic Usage

```bash
python split_pdf_sections.py INPUT.pdf "Label1:StartPage1" "Label2:StartPage2" ...
```

### Example: Splitting a Textbook

```bash
python split_pdf_sections.py textbook.pdf \
    "Foreword:10" "Preface:13" \
    "Ch01_DesigningForTheLearner:20" \
    "Ch02_IntegratedFramework:34" \
    "Ch03_CourseStructure:58" \
    "Ch04_ContentDesign:80" \
    "Ch05_LearningActivities:104" \
    "Ch06_SocialInteractions:122" \
    "Ch07_AssessmentFeedback:142" \
    "Ch08_PuttingItTogether:158"
```

This will generate:
- `01_Foreword.pdf` (pages 10-12)
- `02_Preface.pdf` (pages 13-19)
- `03_Ch01_DesigningForTheLearner.pdf` (pages 20-33)
- `04_Ch02_IntegratedFramework.pdf` (pages 34-57)
- ... and so on

### Example: Simple Report

```bash
python split_pdf_sections.py report.pdf "Intro:1" "Methods:9" "Results:23"
```

Generates:
- `01_Intro.pdf` (pages 1-8)
- `02_Methods.pdf` (pages 9-22)
- `03_Results.pdf` (pages 23-end)

## Input Format

Each section specification should follow the format:
```
"Label:PageNumber"
```

Where:
- **Label**: Any text describing the section (spaces, special characters allowed)
- **PageNumber**: The starting page number as shown in your PDF viewer (1-based)

### Rules

1. **Page numbers start at 1**: Use the page numbers you see in your PDF viewer
2. **Ascending order**: Page numbers must be in strictly ascending order
3. **Valid format**: Must follow `Label:PageNumber` pattern
4. **File safety**: Labels are automatically sanitized for safe filenames

## Output

- Files are named with numbered prefixes: `01_Label.pdf`, `02_Label.pdf`, etc.
- Labels are sanitized: spaces and special characters become underscores
- All files are created in the same directory as the input PDF
- Progress is shown with checkmarks and page ranges

## Error Handling

The script validates inputs and provides clear error messages for:

- **Invalid file**: Input PDF doesn't exist
- **Bad format**: Specifications don't follow `Label:PageNumber` format
- **Invalid pages**: Page numbers less than 1
- **Non-ascending**: Page numbers not in ascending order
- **Missing dependency**: `pypdf` not installed

## Testing

Run the test suite:

```bash
pytest -v
```

Tests cover:
- Basic functionality
- Edge cases (empty sections)
- Input validation
- Metadata preservation
- Label sanitization

## Advanced Usage

### From Script

You can also use the functions programmatically:

```python
from split_pdf_sections import split_by_specs, parse_specs
from pathlib import Path

# Parse specifications
specs = parse_specs(["Intro:1", "Methods:9", "Results:23"])

# Split the PDF
pdf_path = Path("document.pdf")
split_by_specs(pdf_path, specs)
```

### Custom Output Directory

Modify the script to output to a different directory by changing the `out_path` assignment in the `split_by_specs` function.

## Implementation Details

| Feature | Description |
|---------|-------------|
| **Page numbers** | Uses 1-based page numbers (what you see in viewer) |
| **Metadata preservation** | Copies document metadata, XMP data, viewer preferences |
| **Bookmarks** | Preserves bookmarks within each section |
| **Non-destructive** | Original file is never modified |
| **Error handling** | Comprehensive validation with clear error messages |
| **File safety** | Automatic sanitization of labels for safe filenames |

## Troubleshooting

### Common Issues

1. **"Install the dependency first"**: Run `pip install pypdf`
2. **"Input file not found"**: Check the file path and ensure it exists
3. **"Bad spec"**: Ensure specifications follow `Label:PageNumber` format
4. **"Page numbers must be in ascending order"**: Reorder your specifications

### Performance

- Works well with PDFs up to several hundred pages
- Memory usage scales with PDF size
- Processing time depends on PDF complexity and number of sections

## License

This script is provided as-is for educational and practical use. 