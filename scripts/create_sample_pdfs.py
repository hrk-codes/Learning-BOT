from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "documents" / "sample"

SAMPLE_DOCUMENTS = {
    "employee-handbook.pdf": [
        "Employee Handbook - Leave Policy",
        "Full-time employees receive 20 paid vacation days per calendar year.",
        "Employees may carry forward up to 10 unused vacation days into the next year.",
        "Contractors do not receive paid vacation days and cannot carry leave forward.",
        "All leave requests must be submitted through the HR portal.",
    ],
    "product-manual.pdf": [
        "Atlas Sensor Product Manual",
        "The supported operating temperature is between 5 and 40 degrees Celsius.",
        "Operating the device outside this range may reduce measurement accuracy.",
        "The status light flashes amber when the sensor requires calibration.",
    ],
    "internal-api-guide.pdf": [
        "Internal API Guide",
        "Every API request must use OAuth 2.0 bearer-token authentication.",
        "Access tokens are sent in the Authorization HTTP header.",
        "The development rate limit is 120 requests per minute.",
        "Never place client secrets in source code or browser-side JavaScript.",
    ],
}


def create_text_pdf(lines: list[str]) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )

    commands = ["BT", "/F1 12 Tf", "72 740 Td", "16 TL"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.extend([f"({escaped}) Tj", "T*"])
    commands.append("ET")

    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, lines in SAMPLE_DOCUMENTS.items():
        target = SAMPLE_DIR / filename
        target.write_bytes(create_text_pdf(lines))
        print(f"created {target}")


if __name__ == "__main__":
    main()
