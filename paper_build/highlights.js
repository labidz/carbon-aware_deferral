const fs = require("fs"); const H = require("./helpers"); const R = require("./numbers"); const { makeCiter } = require("./refs"); const { D } = H;
const p1 = require("./part1")(R, makeCiter(), H);
const doc = new D.Document({ numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: D.LevelFormat.BULLET, text: "•", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: [new D.Paragraph({ children: [new D.TextRun({ text: "Highlights", bold: true, font: H.FONT, size: 28 })], spacing: { after: 160 } }),
      new D.Paragraph({ children: [new D.TextRun({ text: "Evaluating carbon-aware workload deferral on real forecast vintages: dataset, error structure and learned post-processing", italics: true, font: H.FONT, size: 22 })], spacing: { after: 200 } }),
      ...p1.S.highlights.map((h) => H.BUL(h))] }] });
D.Packer.toBuffer(doc).then((b) => { fs.writeFileSync("/home/claude/paper/highlights.docx", b); console.log("highlights ok", p1.S.highlights.map((h) => h.length)); });
