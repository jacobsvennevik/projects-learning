#!/bin/bash
# PDF Section Splitter - Shell wrapper

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if the script exists
if [ ! -f "split_pdf_sections.py" ]; then
    echo "Error: split_pdf_sections.py not found in current directory."
    exit 1
fi

# Check if pypdf is installed
python3 -c "import pypdf" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: pypdf is not installed. Please run: pip install pypdf"
    exit 1
fi

# Execute the Python script with all arguments
python3 split_pdf_sections.py "$@" 