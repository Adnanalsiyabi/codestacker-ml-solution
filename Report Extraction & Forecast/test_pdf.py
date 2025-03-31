import sys
import os
from police_report_ner import PoliceReportNER, test_with_pdf

def main():
    # Check if the PDF file exists
    pdf_path = "police_crime_report_10.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please ensure the file is in the current directory.")
        sys.exit(1)
    
    # Test the PDF file
    print(f"Testing NER on '{pdf_path}'...")
    test_with_pdf(pdf_path)

if __name__ == "__main__":
    main()
