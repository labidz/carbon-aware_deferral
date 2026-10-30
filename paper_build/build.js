const fs = require("fs");
const H = require("./helpers"); const R = require("./numbers"); const { makeCiter } = require("./refs");
const { D } = H;
const c = makeCiter();
const p1 = require("./part1")(R, c, H);
const p2 = require("./part2")(R, c, H, p1.S);
const p3 = require("./part3")(R, c, H, p1.S);
const p4 = require("./part4")(R, c, H, p1.S);
const refParas = c.list().map((t) => new D.Paragraph({ children: H.runs(t, { size: 19 }), spacing: { after: 70, line: 250 }, indent: { left: 480, hanging: 480 }, alignment: D.AlignmentType.LEFT }));

const children = [...p1.front, ...p1.body, ...p2.data, ...p2.methods, ...p3.res, ...p4.disc, ...p4.concl, ...p4.decl, ...p4.refsHeader, ...refParas, H.PB(), ...p4.appendix];

const doc = new D.Document({
  creator: "", title: "Evaluating carbon-aware workload deferral on real forecast vintages", description: "Manuscript for Sustainable Computing: Informatics and Systems",
  styles: {
    default: { document: { run: { font: H.FONT, size: H.SZ } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: H.FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, font: H.FONT }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, italics: true, font: H.FONT }, paragraph: { spacing: { before: 140, after: 60 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: D.LevelFormat.BULLET, text: "•", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }, { level: 1, format: D.LevelFormat.BULLET, text: "–", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
      { reference: "contrib", levels: [{ level: 0, format: D.LevelFormat.DECIMAL, text: "C%1.", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 420 } } } }] },
      { reference: "num", levels: [{ level: 0, format: D.LevelFormat.DECIMAL, text: "%1.", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }, lineNumbers: { countBy: 1, restart: D.LineNumberRestartFormat.CONTINUOUS } },
    footers: { default: new D.Footer({ children: [new D.Paragraph({ alignment: D.AlignmentType.CENTER, children: [new D.TextRun({ children: [D.PageNumber.CURRENT], font: H.FONT, size: 20 })] })] }) },
    children,
  }],
});
D.Packer.toBuffer(doc).then((buf) => { fs.mkdirSync("/mnt/user-data/outputs", { recursive: true }); fs.writeFileSync("/home/claude/paper/manuscript.docx", buf); console.log("written", buf.length, "bytes; abstract words:", p1.S.abstractWords, "; refs:", c.order.length); });
