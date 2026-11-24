"""Build the PG-MS-LR-LPO technical report as a styled Chinese PDF."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "TECHNICAL_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "PG-MS-LR-LPO-Technical-Report.pdf"
TEMP = ROOT / "tmp" / "pdfs"

INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#64748B")
TEAL = colors.HexColor("#087F8C")
TEAL_LIGHT = colors.HexColor("#E7F3F4")
ORANGE = colors.HexColor("#D97706")
GRID = colors.HexColor("#D7DEE8")
PANEL = colors.HexColor("#F4F7FA")
CODE_BG = colors.HexColor("#F1F5F9")

FIGURES = {
    "### 2.1 总体流程": [
        (
            ROOT / "analysis_plots_clean" / "architecture_diagram.png",
            "图 1. PG-MS-LR-LPO 标准结构。解析平移先验与神经解码结果在输出端融合。",
        )
    ],
    "## 5. 实验结果": [
        (
            ROOT / "analysis_plots_clean" / "performance_comparison.png",
            "图 2. Burgers、diffusion-sorption 和 advection 的跨任务性能对比。",
        )
    ],
    "### 5.3 Advection 参数扫描": [
        (
            ROOT / "analysis_plots_clean" / "parameter_sweep_comparison.png",
            "图 3. 六个 advection beta 下的参数扫描结果。",
        )
    ],
    "### 5.4 残差训练预算": [
        (
            ROOT / "analysis_plots_clean" / "residual_budget_analysis.png",
            "图 4. beta=2 与 beta=4 的残差训练预算和 FNO 阈值。",
        )
    ],
    "### 5.5 β=7 结构消融": [
        (
            ROOT / "analysis_plots_clean" / "beta7_residual_ablation.png",
            "图 5. beta=7 高频残差分支消融。",
        )
    ],
    "### 5.6 二维扩展结果": [
        (
            ROOT / "analysis_plots_2d" / "seed_comparison.png",
            "图 6. 二维 diffusion-reaction 三 seed 三方对比。",
        ),
        (
            ROOT / "analysis_plots_2d_ns" / "seed_comparison.png",
            "图 7. 二维不可压缩 NS 的速度与散度三方对比。",
        ),
    ],
    "### 5.7 一维效率与高监督 DeepONet": [
        (
            ROOT / "analysis_plots_clean" / "deeponet_supervision.png",
            "图 8. Burgers DeepONet 监督点数消融。",
        ),
        (
            ROOT / "analysis_plots_clean" / "efficiency_1d_summary.png",
            "图 9. 一维 PG/FNO/DeepONet 推理效率。",
        ),
    ],
}


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\Deng.ttf")
    bold = Path(r"C:\Windows\Fonts\Dengb.ttf")
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("Chinese DengXian fonts are required.")
    pdfmetrics.registerFont(TTFont("DengXian", str(regular)))
    pdfmetrics.registerFont(TTFont("DengXian-Bold", str(bold)))
    pdfmetrics.registerFontFamily(
        "DengXian",
        normal="DengXian",
        bold="DengXian-Bold",
        italic="DengXian",
        boldItalic="DengXian-Bold",
    )
    return "DengXian", "DengXian-Bold"


def build_styles(font: str, bold: str) -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CoverTitle",
            parent=sample["Title"],
            fontName=bold,
            fontSize=25,
            leading=34,
            alignment=TA_CENTER,
            textColor=INK,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=sample["Normal"],
            fontName=font,
            fontSize=11.5,
            leading=18,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=sample["Normal"],
            fontName=font,
            fontSize=9.2,
            leading=14,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=sample["Heading1"],
            fontName=bold,
            fontSize=16,
            leading=21,
            textColor=TEAL,
            spaceBefore=6,
            spaceAfter=8,
            keepWithNext=True,
            outlineLevel=0,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=sample["Heading2"],
            fontName=bold,
            fontSize=11.5,
            leading=16,
            textColor=INK,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
            outlineLevel=1,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=9.4,
            leading=15,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=9.3,
            leading=14.5,
            leftIndent=12,
            bulletIndent=2,
            textColor=INK,
            spaceAfter=2,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=9.2,
            leading=15,
            leftIndent=12,
            rightIndent=12,
            textColor=colors.HexColor("#334155"),
            backColor=PANEL,
            borderColor=TEAL,
            borderWidth=0,
            borderPadding=7,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#1E293B"),
            backColor=CODE_BG,
            borderColor=GRID,
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=7.1,
            leading=9.2,
            textColor=INK,
            spaceAfter=0,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=sample["BodyText"],
            fontName=bold,
            fontSize=7.2,
            leading=9.3,
            textColor=colors.white,
            spaceAfter=0,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=8,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "toc_title": ParagraphStyle(
            "TOCTitle",
            parent=sample["Heading1"],
            fontName=bold,
            fontSize=19,
            leading=24,
            textColor=INK,
            spaceAfter=12,
        ),
        "toc1": ParagraphStyle(
            "TOC1",
            fontName=bold,
            fontSize=9.4,
            leading=16,
            textColor=INK,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=3,
        ),
        "toc2": ParagraphStyle(
            "TOC2",
            fontName=font,
            fontSize=8.6,
            leading=13.5,
            textColor=MUTED,
            leftIndent=14,
            firstLineIndent=0,
        ),
    }


def inline_markup(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<link href="\2" color="#087F8C">\1</link>',
        text,
    )
    text = re.sub(
        r"&lt;(https?://[^&]+)&gt;",
        r'<link href="\1" color="#087F8C">\1</link>',
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"`([^`]+)`",
        r'<font name="Courier" color="#334155">\1</font>',
        text,
    )
    return text


def image_flowables(
    entries: list[tuple[Path, str]],
    styles: dict[str, ParagraphStyle],
    max_width: float,
) -> list:
    flowables = []
    for path, caption in entries:
        if not path.exists():
            continue
        width, height = ImageReader(str(path)).getSize()
        target_width = min(max_width, width * 0.72)
        target_height = target_width * height / width
        max_height = 120 * mm
        if target_height > max_height:
            target_height = max_height
            target_width = target_height * width / height
        flowables.append(
            KeepTogether(
                [
                    Image(str(path), width=target_width, height=target_height),
                    Paragraph(inline_markup(caption), styles["caption"]),
                ]
            )
        )
    return flowables


def parse_table(
    rows: list[str],
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> Table:
    data = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in cells):
            continue
        data.append(cells)
    if not data:
        raise ValueError("empty table")
    column_count = max(len(row) for row in data)
    for row in data:
        row.extend([""] * (column_count - len(row)))
    max_lengths = [
        max(len(data[row][col]) for row in range(len(data)))
        for col in range(column_count)
    ]
    weights = [max(8, min(34, length)) for length in max_lengths]
    total = sum(weights)
    widths = [available_width * weight / total for weight in weights]
    converted = []
    for row_index, row in enumerate(data):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        converted.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = Table(
        converted,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row_index in range(1, len(converted)):
        if row_index % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row_index), (-1, row_index), PANEL)
            )
    table.setStyle(TableStyle(commands))
    return table


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs) -> None:
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[frame], onPage=self.cover_page),
                PageTemplate(id="body", frames=[frame], onPage=self.body_page),
            ]
        )

    def cover_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(TEAL)
        canvas.rect(0, A4[1] - 14 * mm, A4[0], 14 * mm, fill=1, stroke=0)
        canvas.restoreState()

    def body_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, A4[1] - 16 * mm, A4[0] - 20 * mm, A4[1] - 16 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("DengXian", 7.5)
        canvas.drawString(20 * mm, A4[1] - 12 * mm, "PG-MS-LR-LPO Technical Report")
        canvas.drawRightString(
            A4[0] - 20 * mm,
            12 * mm,
            f"{doc.page}",
        )
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            if style not in {"H1", "H2"}:
                return
            level = 0 if style == "H1" else 1
            text = flowable.getPlainText()
            key = f"heading-{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            if style == "H1":
                self.notify("TOCEntry", (0, text, self.page))


def cover_story(styles: dict[str, ParagraphStyle]) -> list:
    story = [
        Spacer(1, 21 * mm),
        Paragraph("技术报告", styles["cover_meta"]),
        Spacer(1, 6 * mm),
        Paragraph("PG-MS-LR-LPO", styles["title"]),
        Paragraph(
            "Physics-Guided Multi-Scale Low-Rank<br/>"
            "Latent PDE Operator",
            styles["subtitle"],
        ),
        Spacer(1, 9 * mm),
        Paragraph(
            "面向参数化一维 PDE 的物理引导神经算子",
            styles["subtitle"],
        ),
        Spacer(1, 14 * mm),
    ]
    summary = [
        ["研究范围", "PDEBench 一维标量方程"],
        ["主要任务", "Burgers / diffusion-sorption / advection"],
        ["参数网格", "beta = 0.2, 0.7, 1.0, 2.0, 4.0, 7.0"],
        ["标准参数预算", "501,435"],
        ["参数匹配 FNO", "503,165"],
        ["版本日期", "2026-11-24"],
    ]
    table = Table(
        [
            [
                Paragraph(f"<b>{label}</b>", styles["body"]),
                Paragraph(inline_markup(value), styles["body"]),
            ]
            for label, value in summary
        ],
        colWidths=[42 * mm, 105 * mm],
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.8, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 17 * mm),
            Paragraph(
                "代码仓库<br/>"
                '<link href="https://github.com/zhengjiawei-abc/PG-MS-LR-LPO" '
                'color="#087F8C">github.com/zhengjiawei-abc/PG-MS-LR-LPO</link>',
                styles["cover_meta"],
            ),
            Spacer(1, 7 * mm),
            Paragraph(
                "本报告区分已经验证的结果与后续研究计划。"
                "所有数值均来自项目内官方 PDEBench 数据划分和参数匹配实验。",
                styles["cover_meta"],
            ),
        ]
    )
    return story


def markdown_story(
    markdown: str,
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> list:
    lines = markdown.splitlines()
    story = []
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False
    skip_cover = True
    page_break_before = {
        "## 2. 模型结构",
        "## 3. 参数预算",
        "## 5. 实验结果",
        "## 10. 复现入口",
    }
    page_break_before_h2 = {
        "### 5.3 Advection 参数扫描",
        "### 5.4 残差训练预算",
        "### 5.5 β=7 结构消融",
        "### 5.6 二维扩展结果",
    }

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines)
        if text.startswith(">"):
            text = text.lstrip("> ")
            story.append(Paragraph(inline_markup(text), styles["quote"]))
        elif re.match(r"^[-*] ", text):
            text = re.sub(r"^[-*] ", "", text)
            story.append(
                Paragraph(inline_markup(text), styles["bullet"], bulletText="•")
            )
        elif re.match(r"^\d+\. ", text):
            marker, text = text.split(". ", 1)
            story.append(
                Paragraph(
                    inline_markup(text),
                    styles["bullet"],
                    bulletText=f"{marker}.",
                )
            )
        else:
            story.append(Paragraph(inline_markup(text), styles["body"]))
        paragraph_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            story.append(Spacer(1, 3))
            story.append(
                KeepTogether(
                    [parse_table(table_lines, styles, available_width)]
                )
            )
            story.append(Spacer(1, 8))
            table_lines = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            code_text = "\n".join(code_lines)
            story.append(
                XPreformatted(
                    html.escape(code_text),
                    styles["code"],
                )
            )
            code_lines = []

    for line in lines:
        if skip_cover:
            if line.startswith("## 摘要"):
                skip_cover = False
            else:
                continue
        if line.startswith("# "):
            continue

        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("## "):
            flush_paragraph()
            if line.strip() in page_break_before:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(line[3:]), styles["h1"]))
            story.extend(
                image_flowables(
                    FIGURES.get(line.strip(), []),
                    styles,
                    available_width,
                )
            )
            continue
        if line.startswith("### "):
            flush_paragraph()
            if line.strip() in page_break_before_h2:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(line[4:]), styles["h2"]))
            story.extend(
                image_flowables(
                    FIGURES.get(line.strip(), []),
                    styles,
                    available_width,
                )
            )
            continue
        if re.match(r"^[-*] ", line.strip()):
            flush_paragraph()
            item = re.sub(r"^[-*] ", "", line.strip())
            story.append(
                Paragraph(
                    inline_markup(item),
                    styles["bullet"],
                    bulletText="•",
                )
            )
            continue
        if re.match(r"^\d+\. ", line.strip()):
            flush_paragraph()
            marker, item = line.strip().split(". ", 1)
            story.append(
                Paragraph(
                    inline_markup(item),
                    styles["bullet"],
                    bulletText=f"{marker}.",
                )
            )
            continue
        if line.strip() in {"---", "***"}:
            flush_paragraph()
            story.append(Spacer(1, 5))
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    flush_table()
    if in_code:
        flush_code()
    return story


def main() -> None:
    register_fonts()
    font, bold = "DengXian", "DengXian-Bold"
    styles = build_styles(font, bold)
    TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    markdown = SOURCE.read_text(encoding="utf-8")
    doc = ReportDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=21 * mm,
        bottomMargin=18 * mm,
        title="PG-MS-LR-LPO 技术报告",
        author="zhengjiawei-abc",
        subject="Physics-Guided Multi-Scale Low-Rank Latent PDE Operator",
    )
    toc = TableOfContents()
    toc.levelStyles = [styles["toc1"], styles["toc2"]]
    story = cover_story(styles)
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    story.append(Paragraph("目录", styles["toc_title"]))
    story.append(toc)
    story.append(PageBreak())
    story.extend(markdown_story(markdown, styles, doc.width))
    doc.multiBuild(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
