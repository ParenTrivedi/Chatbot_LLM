
import os
import csv
import json
import subprocess
import tempfile
from pathlib import Path


# =====================================================
# SUPPORTED DOCUMENT FORMATS
# =====================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".txt",
    ".csv",
    ".json",
    ".md",
    ".rtf",
    ".odt",
    ".ods",
    ".odp",
    ".html",
    ".htm",
    ".epub"
}


# =====================================================
# TEXT FILE READER
# =====================================================

def read_text_file(path):
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return Path(path).read_text(
            encoding="cp1252",
            errors="replace"
        )


# =====================================================
# PDF READER
# =====================================================

def read_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(path)

    if reader.is_encrypted:
        raise ValueError(
            "Password-protected PDF is not supported."
        )

    content = []

    for page_number, page in enumerate(
        reader.pages, start=1
    ):
        text = page.extract_text() or ""

        if text.strip():
            content.append(
                f"\n[Page {page_number}]\n{text}"
            )

    return "\n".join(content)


# =====================================================
# WORD READER
# =====================================================

def read_docx(path):
    from docx import Document

    document = Document(path)
    content = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            content.append(paragraph.text)

    for table_number, table in enumerate(
        document.tables, start=1
    ):
        content.append(f"\n[Table {table_number}]")

        for row in table.rows:
            values = [
                cell.text.strip()
                for cell in row.cells
            ]

            content.append(" | ".join(values))

    return "\n".join(content)


# =====================================================
# EXCEL XLSX READER
# =====================================================

def read_xlsx(path):
    from openpyxl import load_workbook

    workbook = load_workbook(
        path,
        read_only=True,
        data_only=True
    )

    content = []

    for sheet in workbook.worksheets:
        content.append(
            f"\n[Sheet: {sheet.title}]"
        )

        for row in sheet.iter_rows(values_only=True):
            values = [
                str(value) if value is not None else ""
                for value in row
            ]

            if any(value.strip() for value in values):
                content.append(" | ".join(values))

    workbook.close()

    return "\n".join(content)


# =====================================================
# EXCEL XLS READER
# =====================================================

def read_xls(path):
    import xlrd

    workbook = xlrd.open_workbook(path)
    content = []

    for sheet in workbook.sheets():
        content.append(
            f"\n[Sheet: {sheet.name}]"
        )

        for row_number in range(sheet.nrows):
            values = [
                str(value)
                for value in sheet.row_values(row_number)
            ]

            if any(value.strip() for value in values):
                content.append(" | ".join(values))

    return "\n".join(content)


# =====================================================
# POWERPOINT PPTX READER
# =====================================================

def read_pptx(path):
    from pptx import Presentation

    presentation = Presentation(path)
    content = []

    for slide_number, slide in enumerate(
        presentation.slides, start=1
    ):
        content.append(
            f"\n[Slide {slide_number}]"
        )

        for shape in slide.shapes:

            if shape.has_text_frame:
                text = shape.text.strip()

                if text:
                    content.append(text)

            if shape.has_table:
                for row in shape.table.rows:
                    values = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    content.append(" | ".join(values))

    return "\n".join(content)


# =====================================================
# CSV READER
# =====================================================

def read_csv(path):
    content = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as file:

        sample = file.read(4096)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(file, dialect)

        for row in reader:
            content.append(" | ".join(row))

    return "\n".join(content)


# =====================================================
# JSON READER
# =====================================================

def read_json(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        data = json.load(file)

    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2
    )


# =====================================================
# RTF READER
# =====================================================

def read_rtf(path):
    from striprtf.striprtf import rtf_to_text

    raw_text = read_text_file(path)

    return rtf_to_text(raw_text)


# =====================================================
# HTML READER
# =====================================================

def read_html(path):
    from bs4 import BeautifulSoup

    html = read_text_file(path)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    for element in soup([
        "script",
        "style",
        "noscript"
    ]):
        element.decompose()

    return soup.get_text(
        separator="\n",
        strip=True
    )


# =====================================================
# EPUB READER
# =====================================================

def read_epub(path):
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(path)

    content = []

    for item in book.get_items_of_type(
        ebooklib.ITEM_DOCUMENT
    ):
        soup = BeautifulSoup(
            item.get_content(),
            "html.parser"
        )

        content.append(
            soup.get_text(
                separator="\n",
                strip=True
            )
        )

    return "\n".join(content)


# =====================================================
# ODT READER
# =====================================================

def read_odt(path):
    from odf.opendocument import load
    from odf.text import P, H
    from odf.teletype import extractText

    document = load(path)
    content = []

    for element in (
        document.getElementsByType(H)
        + document.getElementsByType(P)
    ):
        text = extractText(element)

        if text.strip():
            content.append(text)

    return "\n".join(content)


# =====================================================
# ODS READER
# =====================================================

def read_ods(path):
    from odf.opendocument import load
    from odf.table import Table, TableRow
    from odf.teletype import extractText

    document = load(path)
    content = []

    for sheet in document.getElementsByType(Table):

        content.append(
            f"\n[Sheet: {sheet.getAttribute('name')}]"
        )

        for row in sheet.getElementsByType(TableRow):

            values = []

            for cell in row.childNodes:
                if cell.qname[1] == "table-cell":
                    values.append(extractText(cell))

            if any(value.strip() for value in values):
                content.append(" | ".join(values))

    return "\n".join(content)


# =====================================================
# ODP READER
# =====================================================

def read_odp(path):
    from odf.opendocument import load
    from odf.presentation import DrawPage
    from odf.text import P, H
    from odf.teletype import extractText

    document = load(path)
    content = []

    slides = document.getElementsByType(DrawPage)

    for slide_number, slide in enumerate(
        slides, start=1
    ):
        content.append(
            f"\n[Slide {slide_number}]"
        )

        for element in (
            slide.getElementsByType(H)
            + slide.getElementsByType(P)
        ):
            text = extractText(element)

            if text.strip():
                content.append(text)

    return "\n".join(content)


# =====================================================
# LEGACY DOCUMENT CONVERSION
# Requires LibreOffice
# =====================================================

def convert_legacy_document(path, target_format):

    with tempfile.TemporaryDirectory() as temp_dir:

        try:
            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    target_format,
                    "--outdir",
                    temp_dir,
                    str(path)
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120
            )

        except FileNotFoundError:
            raise RuntimeError(
                "LibreOffice is required for DOC and PPT files. "
                "Install LibreOffice and make sure soffice "
                "is available in your system PATH."
            )

        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Document conversion failed: {error.stderr}"
            )

        converted_path = (
            Path(temp_dir)
            / f"{Path(path).stem}.{target_format}"
        )

        if not converted_path.exists():
            raise RuntimeError(
                "LibreOffice did not produce the converted file."
            )

        if target_format == "docx":
            return read_docx(converted_path)

        if target_format == "pptx":
            return read_pptx(converted_path)

        raise ValueError(
            f"Unsupported conversion target: {target_format}"
        )


# =====================================================
# MAIN DOCUMENT EXTRACTION FUNCTION
# =====================================================

def extract_text(file_path):

    path = Path(file_path)

    extension = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            "The provided path is not a file."
        )

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    readers = {
        ".pdf": read_pdf,
        ".docx": read_docx,
        ".xlsx": read_xlsx,
        ".xls": read_xls,
        ".pptx": read_pptx,
        ".txt": read_text_file,
        ".md": read_text_file,
        ".csv": read_csv,
        ".json": read_json,
        ".rtf": read_rtf,
        ".html": read_html,
        ".htm": read_html,
        ".epub": read_epub,
        ".odt": read_odt,
        ".ods": read_ods,
        ".odp": read_odp
    }

    if extension == ".doc":
        extracted = convert_legacy_document(
            path,
            "docx"
        )

    elif extension == ".ppt":
        extracted = convert_legacy_document(
            path,
            "pptx"
        )

    else:
        extracted = readers[extension](path)

    if not extracted.strip():
        raise ValueError(
            "No readable text found. "
            "The document may be empty, image-only or scanned."
        )

    return extracted


# =====================================================
# DOCUMENT INFORMATION FUNCTION
# =====================================================

def get_document_info(file_path):

    path = Path(file_path)

    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": os.path.getsize(path),
        "supported": (
            path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    }
