# convert_docx_to_md.py
import os
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

def docx_to_markdown(docx_path, md_path):
    if not os.path.exists(docx_path):
        print(f"Error: Docx file not found at {docx_path}")
        return

    doc = Document(docx_path)
    body_elements = doc.element.body
    
    markdown_lines = []
    image_counter = 0
    image_names = [
        "loss_history.png",
        "predictions.png",
        "scatter.png",
        "residuals.png"
    ]
    
    # We need to map XML elements to python-docx Paragraph or Table objects
    # to iterate through them in document order.
    for child in body_elements:
        tag = child.tag
        if tag.endswith('p'):  # Paragraph
            p = Paragraph(child, doc)
            text = p.text.strip()
            style_name = p.style.name
            
            # Check for images in this paragraph
            has_image = False
            if 'w:drawing' in child.xml or 'pic:pic' in child.xml:
                has_image = True
            
            if style_name.startswith('Heading '):
                level = int(style_name.split()[-1])
                markdown_lines.append(f"\n{'#' * level} {text}\n")
            elif style_name == 'Title':
                markdown_lines.append(f"\n# {text}\n")
            elif style_name == 'Subtitle':
                markdown_lines.append(f"\n### {text}\n")
            elif style_name == 'List Bullet':
                # Reconstruct bold/italic formatting inside lists if any
                formatted_text = ""
                for run in p.runs:
                    run_text = run.text
                    if not run_text:
                        continue
                    if run.bold:
                        formatted_text += f"**{run_text}**"
                    elif run.italic:
                        formatted_text += f"*{run_text}*"
                    else:
                        formatted_text += run_text
                markdown_lines.append(f"* {formatted_text.strip()}")
            elif has_image:
                if image_counter < len(image_names):
                    img_name = image_names[image_counter]
                    caption = img_name.replace(".png", "").replace("_", " ").title()
                    markdown_lines.append(f"\n![{caption}](./output/{img_name})\n")
                    image_counter += 1
            else:
                if text:
                    # Reconstruct bold/italic formatting inside normal paragraphs
                    formatted_text = ""
                    for run in p.runs:
                        run_text = run.text
                        if not run_text:
                            continue
                        if run.bold:
                            formatted_text += f"**{run_text}**"
                        elif run.italic:
                            formatted_text += f"*{run_text}*"
                        else:
                            formatted_text += run_text
                    markdown_lines.append(f"\n{formatted_text.strip()}\n")
                else:
                    # Empty paragraph, skip to keep formatting clean
                    pass
                    
        elif tag.endswith('tbl'):  # Table
            tbl = Table(child, doc)
            md_table_rows = []
            
            # Find the max number of columns in any row to make sure it's regular
            num_cols = max(len(row.cells) for row in tbl.rows) if tbl.rows else 0
            if num_cols == 0:
                continue
                
            for i, row in enumerate(tbl.rows):
                row_cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                
                # Skip completely empty rows
                if all(not c.strip() for c in row_cells):
                    continue
                    
                cleaned_cells = []
                for idx, c_text in enumerate(row_cells):
                    cleaned_cells.append(c_text)
                    
                md_table_rows.append("| " + " | ".join(cleaned_cells) + " |")
                if i == 0:
                    md_table_rows.append("| " + " | ".join(["---"] * len(cleaned_cells)) + " |")
            
            markdown_lines.append("\n" + "\n".join(md_table_rows) + "\n")
            
    # Combine lines to output
    content = ""
    for line in markdown_lines:
        content += line + "\n"
        
    # Standardize spaces and consecutive newlines
    while "\n\n\n" in content:
        content = content.replace("\n\n\n", "\n\n")
        
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content.strip() + "\n")
        
    print(f"Successfully converted {docx_path} to {md_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docx_file = os.path.join(base_dir, 'Raport_Deep_Learning_NYSE_JPM.docx')
    md_file = os.path.join(base_dir, 'README.md')
    docx_to_markdown(docx_file, md_file)
