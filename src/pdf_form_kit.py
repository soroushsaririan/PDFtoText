import random
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = LETTER

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

HAND_FONTS = {
    "caveat": str(FONTS_DIR / "Caveat.ttf"),
    "shadows": str(FONTS_DIR / "ShadowsIntoLight.ttf"),
    "sacramento": str(FONTS_DIR / "Sacramento.ttf"),
}

_REGISTERED = set()


def _register(font_key):
    if font_key in _REGISTERED:
        return font_key
    path = HAND_FONTS[font_key]
    try:
        pdfmetrics.registerFont(TTFont(font_key, path))
    except Exception:
        pdfmetrics.registerFont(TTFont(font_key, path, subfontIndex=0))
    _REGISTERED.add(font_key)
    return font_key


class FormPage:
    def __init__(self, path, hand_font="caveat", seed=0):
        self.path = path
        self.c = canvas.Canvas(str(path), pagesize=LETTER)
        self.hand_font = _register(hand_font)
        self.rng = random.Random(seed)
        self.y = PAGE_H - 55
        self.margin = 50

    def _advance(self, dy):
        self.y -= dy
        if self.y < 60:
            self.new_page()

    def new_page(self):
        self.c.showPage()
        self.y = PAGE_H - 55

    def title(self, text, size=13):
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(self.margin, self.y, text)
        self._advance(size + 12)

    def subtitle(self, text, size=10.5):
        self.c.setFont("Helvetica-Oblique", size)
        self.c.drawString(self.margin, self.y, text)
        self._advance(size + 10)

    def section(self, text, size=11):
        self._advance(6)
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(self.margin, self.y, text)
        self.c.line(self.margin, self.y - 3, PAGE_W - self.margin, self.y - 3)
        self._advance(size + 10)

    def _draw_handwritten(self, text, x, y, size=13, jitter=1.3, rot=2.5, messiness=1.0):
        self.c.setFont(self.hand_font, size)
        cx = x
        drift = 0.0
        for ch in text:
            self.c.saveState()
            dx = self.rng.uniform(-jitter, jitter) * messiness
            dy = self.rng.uniform(-jitter, jitter) * messiness + drift
            dr = self.rng.uniform(-rot, rot) * messiness
            self.c.translate(cx + dx, y + dy)
            self.c.rotate(dr)
            self.c.drawString(0, 0, ch)
            self.c.restoreState()
            if messiness > 1.5:
                spacing = self.rng.uniform(0.55, 1.3)
                drift = max(-4, min(4, drift + self.rng.uniform(-0.7, 0.7)))
            else:
                spacing = self.rng.uniform(0.92, 1.05)
            cx += self.c.stringWidth(ch, self.hand_font, size) * spacing
        return cx

    def field(self, label, answer="", size=10, hand_size=13, answer_x=None, messiness=1.0):
        self.c.setFont("Helvetica", size)
        self.c.drawString(self.margin, self.y, label)
        ax = answer_x or (self.margin + max(230, self.c.stringWidth(label, "Helvetica", size) + 20))
        self.c.setDash(1, 2)
        self.c.line(ax, self.y - 1, PAGE_W - self.margin, self.y - 1)
        self.c.setDash()
        if answer:
            self._draw_handwritten(answer, ax + 4, self.y + 2, size=hand_size, messiness=messiness)
        self._advance(max(size, hand_size) + 12)

    def scratch_out(self, x0, y0, x1, y1, passes=5):
        self.c.saveState()
        self.c.setLineWidth(1.3)
        self.c.setStrokeColorRGB(0.05, 0.05, 0.05)
        for _ in range(passes):
            y_mid = self.rng.uniform(y0, y1)
            points = []
            n = 6
            for i in range(n):
                px = x0 + (x1 - x0) * i / (n - 1)
                py = y_mid + self.rng.uniform(-(y1 - y0) / 2, (y1 - y0) / 2)
                points.append((px, py))
            p = self.c.beginPath()
            p.moveTo(*points[0])
            for pt in points[1:]:
                p.lineTo(*pt)
            self.c.drawPath(p)
        self.c.restoreState()

    def scratched_field(self, label, wrong_answer, correct_answer, size=10, hand_size=13, messiness=1.0):
        self.c.setFont("Helvetica", size)
        self.c.drawString(self.margin, self.y, label)
        ax = self.margin + max(230, self.c.stringWidth(label, "Helvetica", size) + 20)
        self.c.setDash(1, 2)
        self.c.line(ax, self.y - 1, PAGE_W - self.margin, self.y - 1)
        self.c.setDash()
        wrong_end_x = self._draw_handwritten(wrong_answer, ax + 4, self.y + 2, size=hand_size, messiness=messiness)
        self.scratch_out(ax + 2, self.y - 2, wrong_end_x + 4, self.y + hand_size + 2)
        self._draw_handwritten(correct_answer, wrong_end_x + 45, self.y + 2, size=hand_size, messiness=messiness)
        self._advance(max(size, hand_size) + 12)

    def yes_no(self, label, value, size=10):
        self.c.setFont("Helvetica", size)
        self.c.drawString(self.margin, self.y, label)
        yes_x, no_x = self.margin + 260, self.margin + 320
        self.c.drawString(yes_x, self.y, "Yes")
        self.c.drawString(no_x, self.y, "No")
        cx = yes_x - 4 if value == "Yes" else no_x - 4
        cw = self.c.stringWidth("Yes" if value == "Yes" else "No", "Helvetica", size)
        self.c.saveState()
        self.c.setStrokeColorRGB(0.1, 0.1, 0.6)
        self.c.setLineWidth(1.4)
        self.c.ellipse(cx - 3, self.y - 3, cx + cw + 5, self.y + size + 1, stroke=1, fill=0)
        self.c.restoreState()
        self._advance(size + 12)

    def checkboxes(self, label, options, selected, size=10, hand_size=12, messiness=1.0):
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(self.margin, self.y, label)
        self._advance(size + 8)
        col_w = (PAGE_W - 2 * self.margin) / 2
        for i, opt in enumerate(options):
            col = i % 2
            row = i // 2
            x = self.margin + col * col_w
            y = self.y - row * (size + 8)
            self.c.rect(x, y - 2, size, size, stroke=1, fill=0)
            if opt in selected:
                self._draw_handwritten("X", x + 1, y - 1, size=size + 1, messiness=messiness)
            self.c.setFont("Helvetica", size)
            self.c.drawString(x + size + 4, y, opt)
        rows = (len(options) + 1) // 2
        self._advance(rows * (size + 8) + 6)

    def note(self, label, text, size=10, hand_size=12, lines=2, messiness=1.0):
        self.c.setFont("Helvetica", size)
        self.c.drawString(self.margin, self.y, label)
        self._advance(size + 8)
        words = text.split()
        line, cur_w, max_w = [], 0, PAGE_W - 2 * self.margin - 10
        drawn_lines = []
        for w in words:
            wlen = len(w) * hand_size * 0.55
            if cur_w + wlen > max_w:
                drawn_lines.append(" ".join(line))
                line, cur_w = [], 0
            line.append(w)
            cur_w += wlen + hand_size * 0.3
        if line:
            drawn_lines.append(" ".join(line))
        for ln in drawn_lines[:lines] or [""]:
            self.c.setDash(1, 2)
            self.c.line(self.margin, self.y - 1, PAGE_W - self.margin, self.y - 1)
            self.c.setDash()
            self._draw_handwritten(ln, self.margin + 4, self.y + 2, size=hand_size, messiness=messiness)
            self._advance(hand_size + 10)

    def save(self):
        self.c.showPage()
        self.c.save()
