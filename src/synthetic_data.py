"""
Synthetic table image generator.

Creates a simple table image with known text content so the demo is
fully self-contained and deterministic — no external image files needed.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Optional


def generate_sample_table_image(
    save_path: str = "outputs/sample_table.png",
    width: int = 600,
    height: int = 400,
) -> str:
    """
    Generate a synthetic table image with known text cells.

    The table looks like:
        Item   | Price | Qty
        Apple  | $2    | 5
        Banana | $1    | 12
        Orange | $3    | 8

    Args:
        save_path: where to save the generated image
        width: image width in pixels
        height: image height in pixels

    Returns:
        The path to the saved image.
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Try to use a readable font; fall back to default if unavailable
    font = _get_font(size=22)
    header_font = _get_font(size=24)

    # Table data
    headers = ["Item", "Price", "Qty"]
    rows = [
        ["Apple", "$2", "5"],
        ["Banana", "$1", "12"],
        ["Orange", "$3", "8"],
    ]

    # Layout constants
    margin_x, margin_y = 60, 60
    col_width = 160
    row_height = 60
    num_cols = len(headers)
    num_rows = len(rows) + 1  # +1 for header

    table_w = col_width * num_cols
    table_h = row_height * num_rows

    # Draw outer border
    draw.rectangle(
        [margin_x, margin_y, margin_x + table_w, margin_y + table_h],
        outline="black",
        width=2,
    )

    # Draw rows and columns, fill cells with text
    for row_idx in range(num_rows):
        y = margin_y + row_idx * row_height

        # Horizontal line
        if row_idx > 0:
            draw.line([(margin_x, y), (margin_x + table_w, y)], fill="black", width=2)

        for col_idx in range(num_cols):
            x = margin_x + col_idx * col_width

            # Vertical line
            if col_idx > 0:
                draw.line(
                    [(x, margin_y), (x, margin_y + table_h)], fill="black", width=2
                )

            # Determine cell text
            if row_idx == 0:
                text = headers[col_idx]
                f = header_font
            else:
                text = rows[row_idx - 1][col_idx]
                f = font

            # Center text in cell
            bbox = draw.textbbox((0, 0), text, font=f)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x + (col_width - tw) // 2
            ty = y + (row_height - th) // 2
            draw.text((tx, ty), text, fill="black", font=f)

    # Add a title above the table
    title = "Product Inventory"
    title_font = _get_font(size=28)
    tbbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (tbbox[2] - tbbox[0])) // 2, 15),
        title,
        fill="black",
        font=title_font,
    )

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
    print(f"[synthetic_data] Saved sample table image to {save_path}")
    return save_path


def _get_font(size: int = 20) -> ImageFont.FreeTypeFont:
    """Try common system font paths; fall back to default bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # Last resort: default font (small but works everywhere)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    path = generate_sample_table_image()
    img = Image.open(path)
    print(f"  Image size: {img.size}")
    print(f"  Image mode: {img.mode}")
    img.show()
