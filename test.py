import json
import os
from fpdf import FPDF

# Function to create PDF from resume content
def create_pdf(content, output_filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add a font that supports Unicode characters (e.g., Arial)
    pdf.add_font('Arial', '', 'C:/Windows/Fonts/arial.ttf', uni=True)
    pdf.set_font('Arial', size=12)
    pdf.multi_cell(0, 10, content)

    pdf.output(output_filename)

# Function to process the JSONL file and create PDFs
def process_resumes(input_filename):
    # Create the output directory if it doesn't exist
    output_directory = 'generated_resumes'
    os.makedirs(output_directory, exist_ok=True)

    with open(input_filename, 'r', encoding='utf-8') as file:
        line_number = 1
        for line in file:
            resume = json.loads(line)
            content = resume.get('content', '')
            if content:
                # Generate PDF file with unique name inside the desired folder
                output_filename = os.path.join(output_directory, f"resume_{line_number}.pdf")
                create_pdf(content, output_filename)
                print(f"Created: {output_filename}")
            line_number += 1

# Set the path to your JSONL file
input_filename = 'cleaned_data.jsonl'  # Replace with the actual path to your JSONL file

# Process resumes and create PDFs
process_resumes(input_filename)
