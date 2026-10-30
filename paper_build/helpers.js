const fs = require("fs");
const D = require("docx");
const { Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType, ShadingType, HeadingLevel, LevelFormat } = D;

const FONT = "Times New Roman";
const SZ = 22; // 11 pt
const CONTENT_W = 9026; // A4 with 2.54 cm margins = 11906 - 2*1440 = 9026 DXA

// ---- inline markup: **bold**, *italic*, ^sup^, ~sub~ ----
function runs(text, base = {}) {
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)/g; let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: SZ, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, size: SZ, ...base }));
    else if (tok.startsWith("*")) out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, size: SZ, ...base }));
    else if (tok.startsWith("^")) out.push(new TextRun({ text: tok.slice(1, -1), superScript: true, font: FONT, size: SZ, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), subScript: true, font: FONT, size: SZ, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: SZ, ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), alignment: opts.align || AlignmentType.JUSTIFIED, spacing: { after: opts.after ?? 120, line: opts.line ?? 300 }, indent: opts.indent, keepNext: opts.keepNext, ...(opts.p || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, bold: true, font: FONT, size: 26 })], spacing: { before: 280, after: 120 }, keepNext: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, bold: true, font: FONT, size: 23 })], spacing: { before: 200, after: 80 }, keepNext: true });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, bold: true, italics: true, font: FONT, size: SZ })], spacing: { before: 140, after: 60 }, keepNext: true });
const BUL = (text, level = 0) => new Paragraph({ numbering: { reference: "bul", level }, children: runs(text), spacing: { after: 60, line: 290 }, alignment: AlignmentType.JUSTIFIED });
const NUM = (text, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text), spacing: { after: 60, line: 290 }, alignment: AlignmentType.JUSTIFIED });

// equation: centred text with right-aligned number via tab stop
function EQ(text, num) {
  return new Paragraph({
    tabStops: [{ type: D.TabStopType.CENTER, position: CONTENT_W / 2 }, { type: D.TabStopType.RIGHT, position: CONTENT_W }],
    children: [new TextRun({ text: "\t", font: FONT, size: SZ }), ...runs(text, { }), new TextRun({ text: "\t(" + num + ")", font: FONT, size: SZ })],
    spacing: { before: 80, after: 120 },
  });
}

// figure: image + caption
function FIG(path, caption, widthCm = 16, aspect = null) {
  const buf = fs.readFileSync(path);
  // read PNG dimensions
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wEmu = Math.round(widthCm * 360000), pxW = Math.round(widthCm / 2.54 * 96), pxH = Math.round(pxW * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true, children: [new ImageRun({ type: "png", data: buf, transformation: { width: pxW, height: pxH }, altText: { title: caption.slice(0, 40), description: caption, name: "figure" } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200, line: 260 }, children: runs(caption, { size: 19 }) }),
  ];
}

// three-line (booktabs-style) table
function TABLE(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const rule = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
  const cell = (t, i, r, isHead, isLast, bold = false) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: isHead && r === 0 ? rule(8) : none, bottom: isHead ? rule(4) : (isLast ? rule(8) : none), left: none, right: none },
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    verticalAlign: D.VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : (opts.align || AlignmentType.CENTER), children: runs(String(t), { size: opts.size || 18, bold: isHead || bold }), spacing: { after: 0, line: 240 } })],
  });
  const trs = [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, 0, true, false)) })];
  rows.forEach((row, r) => {
    const isSep = row.length === 1 && String(row[0]).startsWith("#");
    if (isSep) { trs.push(new TableRow({ children: [new TableCell({ columnSpan: widths.length, width: { size: total, type: WidthType.DXA }, borders: { top: none, bottom: none, left: none, right: none }, margins: { top: 60, bottom: 20, left: 70, right: 70 }, children: [new Paragraph({ children: runs(String(row[0]).slice(1), { size: opts.size || 18, italics: true, bold: true }), spacing: { after: 0 } })] })] })); return; }
    trs.push(new TableRow({ cantSplit: true, children: row.map((t, i) => cell(t, i, r + 1, false, r === rows.length - 1, opts.boldRows && opts.boldRows.includes(r))) }));
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: trs, alignment: AlignmentType.CENTER });
}
const CAP = (t) => new Paragraph({ keepNext: true, spacing: { before: 160, after: 60, line: 260 }, alignment: AlignmentType.JUSTIFIED, children: runs(t, { size: 19 }) });
const NOTE = (t) => new Paragraph({ spacing: { before: 40, after: 200, line: 240 }, alignment: AlignmentType.JUSTIFIED, children: runs(t, { size: 17 }) });
const SPACER = () => new Paragraph({ children: [], spacing: { after: 60 } });
const PB = () => new Paragraph({ children: [new D.PageBreak()] });

// number formatting
const f = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : Number(x).toFixed(d);
const pct = (x, d = 1) => f(x, d) + "%";
const ci = (a, lo, hi, d = 1) => `${f(a, d)} [${f(lo, d)}, ${f(hi, d)}]`;
const sgn = (x, d = 1) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const minus = (s) => String(s).replace(/-(?=\d)/g, "−");

module.exports = { D, FONT, SZ, CONTENT_W, runs, P, H1, H2, H3, BUL, NUM, EQ, FIG, TABLE, CAP, NOTE, SPACER, PB, f, pct, ci, sgn, minus };
